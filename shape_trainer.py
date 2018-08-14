import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data_loader import PouringDataset
from base_trainer import BaseTrainer

from tqdm import tqdm
tqdm.monitor_interval = 0

from models import ConvNet

class ShapeTrainer(BaseTrainer):

    def _init_fn(self):

        if self.options.task=='wait_times':
            self.use_speed_and_angle = True
        else:
            self.use_speed_and_angle = False

        self.model = ConvNet(input_image_size=self.options.image_size,
                             num_output_channels=self.options.num_horz_divs,
                             num_hidden_channels=self.options.num_hidden_channels,
                             num_linear_layers=self.options.num_hidden_layers,
                             dropout_prob=self.options.dropout,
                             use_speed_and_angle=self.use_speed_and_angle,
                             nonlinearity=self.options.nonlinearity).to(self.device)


        self.train_ds = PouringDataset(self.options.dataset_dir,
                                       load_volume=self.options.task!='cross_section',
                                       calc_wait_times=self.options.task=='wait_times',
                                       load_speed_angle_and_scale=self.use_speed_and_angle,
                                       volume_dir=self.options.volume_dir,
                                       num_divisions=self.options.num_horz_divs,
                                       image_size=self.options.image_size,
                                       center=self.options.center,
                                       is_train=True)
        self.test_ds = PouringDataset(self.options.dataset_dir,
                                      load_volume=self.options.task!='cross_section',
                                      calc_wait_times=self.options.task=='wait_times',
                                      load_speed_angle_and_scale=self.use_speed_and_angle,
                                      volume_dir=self.options.volume_dir,
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
        if is_train:
            input_batch = [input_batch]
            pred_profiles = [pred_profiles]

        self.summary_writer.add_scalar(prefix + 'loss', loss, self.step_count)

        profile_images = []
        for j in range(len(pred_profiles)):
            for i in range(len(pred_profiles[j])):
                if self.options.task == 'cross_section':
                    gt_profile = input_batch[j]['cross_section_profile'][i]
                elif self.options.task == 'volume_profile':
                    gt_profile = input_batch[j]['volume_profile'][i]
                elif self.options.task == 'wait_times':
                    gt_profile = input_batch[j]['wait_times'][i]
                else:
                    raise NotImplementedError('The requested task does not exist') 

                profile_image = self._make_profile_image(gt_profile, pred_profiles[j][i])
                profile_image = profile_image.to(self.device, dtype=torch.float64)

                resized_image = cv2.resize(input_batch[j]['depth_image'][i].cpu().numpy(), (profile_image.shape[1], profile_image.shape[2]))
                resized_image = torch.from_numpy(resized_image)
                color_image = torch.zeros_like(profile_image)
                color_image[0, :, :] = resized_image
                color_image[1, :, :] = resized_image
                color_image[2, :, :] = resized_image

                profile_images.append(color_image)
                profile_images.append(profile_image)


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

            image[0, len(gt_profile) - i - 1, gt_index] = 1

            output_index = max(0, output_index)
            output_index = min(width-1, output_index)
            image[1, len(gt_profile) - i - 1, output_index] = 1
        return image

    def test(self):
        test_data_loader = DataLoader(self.test_ds, batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test)
        all_profiles = []
        all_batches = []
        test_loss = torch.tensor(0.0, device=self.device)

        for tstep, batch in enumerate(tqdm(test_data_loader, desc='Testing')):
            batch = {k: v.to(self.device) for k,v in batch.items()}

            pred_profiles, loss = self._test_step(batch)

            test_loss += loss.data
            all_profiles.append(pred_profiles)
            all_batches.append(batch)

        self._train_summaries(all_batches, all_profiles, test_loss, is_train=False)


        
    def _test_step(self, input_batch):
        self.model.eval()
        return self._train_or_test_step(input_batch, False)


    def _train_or_test_step(self, input_batch, is_train):
        depth_images = input_batch['depth_image'].to(torch.float)
        depth_images = depth_images.view(-1, 1, depth_images.shape[1], depth_images.shape[2])

        if self.options.task == 'cross_section':
            gt_profiles = input_batch['cross_section_profile'].to(torch.float)
        elif self.options.task == 'volume_profile':
            gt_profiles = input_batch['volume_profile'].to(torch.float)
        elif self.options.task == 'wait_times':
            gt_profiles = input_batch['wait_times'].to(torch.float)
        else:
            raise NotImplementedError('The requested task does not exist') 


        if self.use_speed_and_angle:
            speed = input_batch['speed'].to(torch.float)
            angle = input_batch['angle'].to(torch.float)
            with torch.set_grad_enabled(is_train):
                pred_profiles = self.model(depth_images, speed, angle)
        else:
            with torch.set_grad_enabled(is_train):
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
