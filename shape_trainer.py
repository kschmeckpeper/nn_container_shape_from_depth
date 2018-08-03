import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data_loader import PouringDataset
from base_trainer import BaseTrainer

from tqdm import tqdm
tqdm.monitor_interval = 0

from models import ConvNet

class ShapeTrainer(BaseTrainer):

    def _init_fn(self):

        self.model = ConvNet(input_image_size=self.options.image_size,
                             num_output_channels=self.options.num_horz_divs,
                             num_hidden_channels=self.options.num_hidden_channels,
                             num_linear_layers=self.options.num_hidden_layers,
                             dropout_prob=self.options.dropout,
                             nonlinearity=self.options.nonlinearity).to(self.device)


        self.train_ds = PouringDataset(self.options.dataset_dir, 
                                       num_divisions=self.options.num_horz_divs,
                                       image_size=self.options.image_size,
                                       center=self.options.center,
                                       is_train=True)
        self.test_ds = PouringDataset(self.options.dataset_dir,
                                      num_divisions = self.options.num_horz_divs,
                                      image_size=self.options.image_size,
                                      center=self.options.center,
                                      is_train=False)

        if self.options.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.options.lr, momentum=self.options.sgd_momentum, weight_decay=self.options.wd)
        elif self.options.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=self.options.lr, momentum=0, weight_decay=self.options.wd)
        else:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.options.lr, betas=(self.options.adam_beta1, 0.999), weight_decay=self.options.wd)

        # pack all models and optimizers in dictionaries to interact with the checkpoint saver
        self.models_dict = {'stacked_hg': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        if self.options.loss == 'L1':
            self.criterion = nn.L1Loss(size_average=True).to(self.device)
        elif self.options.loss == 'SmoothL1':
            self.criterion = nn.SmoothL1Loss(size_average=True).to(self.device)
        else:
            self.criterion = nn.MSELoss(size_average=True).to(self.device)


    def _train_step(self, input_batch):
        self.model.train()
        return self._train_or_test_step(input_batch, True)



    def _train_summaries(self, input_batch, pred_profiles, loss, is_train=True):
        prefix = 'train/' if is_train else 'test/'
        self.summary_writer.add_scalar(prefix + 'loss', loss, self.step_count)

        profile_images = []
        for i in range(len(pred_profiles)):
            profile_images.append(input_batch['depth_image'][i])
            profile_images.append(self._make_profile_image(input_batch['profile'][i], pred_profiles[i]))


        profile_image_grid = make_grid(profile_images, pad_value=1, nrow=4)
        self.summary_writer.add_image(prefix + 'profiles', profile_image_grid, self.step_count)
        if is_train:
            self.summary_writer.add_scalar('lr', self._get_lr(), self.step_count)

    def _make_profile_image(self, gt_profile, output_profile):
        gt_profile = gt_profile.to(torch.float)
        width = max(len(gt_profile), 128)
        image = torch.zeros((3, len(gt_profile), width))
        
        max_gt = gt_profile.max() * 1.5
        max_img = output_profile.max() * 1.1
        max_gt = max(max_gt, max_img)
        max_gt = 1.5
        #print output_profile.max(), output_profile.min()

        for i in range(len(gt_profile)):
            gt_index = int(1.0 * width * gt_profile[i] / max_gt)
            output_index = int(1.0 * width * output_profile[i] / max_gt)

            image[0, i, gt_index] = 1

            output_index = max(0, output_index)
            output_index = min(width-1, output_index)
            image[1, i, output_index] = 1
        return image

    def test(self):
        test_data_loader = DataLoader(self.test_ds, batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test)
        profiles = None
        last_batch = None
        test_loss = torch.tensor(0.0, device=self.device)

        for tstep, batch in enumerate(tqdm(test_data_loader, desc='Testing')):
            batch = {k: v.to(self.device) for k,v in batch.items()}

            pred_profiles, loss = self._test_step(batch)

            test_loss += loss.data
            profiles = pred_profiles
            last_batch = batch

        self._train_summaries(last_batch, profiles, test_loss, is_train=False)


        
    def _test_step(self, input_batch):
        self.model.eval()
        return self._train_or_test_step(input_batch, False)


    def _train_or_test_step(self, input_batch, is_train):
        depth_images = input_batch['depth_image'].to(torch.float)
        depth_images = depth_images.view(-1, 1, depth_images.shape[1], depth_images.shape[2])
        gt_profiles = input_batch['profile'].to(torch.float)

        # Turn off gradients when testing
        if is_train:
            pred_profiles = self.model(depth_images)
        else:
            with torch.no_grad():
                pred_profiles = self.model(depth_images)


        loss = torch.tensor(0.0, device=self.device).to(torch.float)
        for i in range(len(pred_profiles)):
            loss += self.criterion(pred_profiles[i], gt_profiles[i])

        # Only do backprop when training
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return [profile.detach() for profile in pred_profiles], loss.detach()
