import numpy as np
from torch.utils.data import Dataset
import os
from enum import Enum
import torch
import cv2

class Shape(Enum):
    STRAIGHT=0
    FLASK=1
    SINUSOID=2



class PouringDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 is_train,
                 load_volume=False,
                 volume_dir=None,
                 num_divisions=128,
                 image_size=128,
                 center=False,
                 calc_wait_times=False,
                 load_speed_angle_and_scale=False):

        self.num_divisions = num_divisions
        self.root_dir = root_dir
        self.center = center
        self.image_size = image_size
        self.load_volume = load_volume
        self.volume_dir = volume_dir
        self.calc_wait_times = calc_wait_times
        self.load_speed_angle_and_scale = load_speed_angle_and_scale

        if is_train:
            file_path = os.path.join(root_dir, "train.txt")
        else:
            file_path = os.path.join(root_dir, "test.txt")

        with open(file_path, 'r') as data_file:
            self.files = data_file.read().splitlines()

    def __len__(self):
        return len(self.files)


    def _get_container_profile(self, cfg_file_path):
        with open(cfg_file_path, 'r') as cfg_file:
            lines = cfg_file.read().splitlines()

            line_count = len(lines)

            if line_count == 6:
                shape = Shape.STRAIGHT
            elif line_count == 7:
                shape = Shape.FLASK
            elif line_count == 8: #Sinusoidal
                shape = Shape.SINUSOID
            else:
                print "Invalid cfg file format"

            neck_radius = 0
            base_radius = 0
            a = 0
            b = 0
            c = 0
            d = 0

            for line in lines:
                split = line.split()
                # print split
                if split[0] == 'neck_radius:':
                    neck_radius = float(split[1])
                elif split[0] == 'base_radius:':
                    base_radius = float(split[1])
                elif len(split) == 9 and split[7] == 'da:':
                    a = float(split[8])
                elif split[0] == 'b:':
                    b = float(split[1])
                elif split[0] == 'c:':
                    c = float(split[1])
                elif split[0] == 'd:':
                    d = float(split[1])
                elif split[0] == 'neck_height:':
                    neck_height = float(split[1])
                elif split[0] == 'Radius':
                    # Hack because the flask cfgs were not correctly formatted
                    base_radius = float(split[-1])

            profile = np.zeros(self.num_divisions)

            if shape == Shape.STRAIGHT:
                profile = np.linspace(base_radius, neck_radius, self.num_divisions)

            elif shape == Shape.SINUSOID:
                profile = a * np.sin( b * np.linspace(0, 1, self.num_divisions) + c) + d

            elif shape == Shape.FLASK:
                split = int(self.num_divisions * (1 - neck_height))
                profile[:split] = np.linspace(base_radius, neck_radius, split)
                profile[split:] = np.linspace(neck_radius, neck_radius, self.num_divisions - split)
            else:
                print "Invalid cfg file format"
            return profile

    def _get_volume_profile(self, data):
        
        max_volume = data[0, 1]
        # Flip volume to make it the amount in the receiving container
        # instead of the amount in the pouring container
        volumes = (max_volume - data[:, 0]) / 100000.0

        step_size = len(volumes) / self.num_divisions
        volume_profile = np.zeros(self.num_divisions)

        for i in range(self.num_divisions):
            volume_profile[i] = np.mean(volumes[i * step_size:(i+1)*step_size])

        return volume_profile

    def _calc_wait_times(self, volume_data, threshold_fraction=0.75):
        eps = 0.000001

        angles = volume_data[:, 2]

        double_diff = np.diff(np.diff(angles))
        start_indices = np.where(double_diff < -eps)
        end_indices = np.where(double_diff > eps)

        start_volumes = volume_data[start_indices, 0]
        end_volumes = volume_data[end_indices, 0]

        volume_differences = start_volumes - end_volumes
        volume_thresholds = end_volumes + (1.0 - threshold_fraction) * volume_differences

        volume_index = 0
        wait_times = []
        for i in range(len(volume_data)):
            if volume_data[i, 0] > volume_thresholds[volume_index]:
                wait_times.append(volume_data[i, 3] - volume_data[start_indices, 3])
                volume_index += 1
                if volume_index >= len(volume_thresholds):
                    return wait_times

        return wait_times


    def _load_params(self, base_file_path):
        file_path = base_file_path.split('.')[0] + '_params.txt'

        with open(file_path, 'r') as param_file:
            for line in param_file.readlines():
                split = line.split(' ')
                if split[0] == 'rotation_speed':
                    speed = float(split[1])
                elif split[0] == 'stop_angle':
                    angle = float(split[1])
                elif split[0] == 'scaling_factor':
                    scaling_factor = float(split[1])
        return speed, angle, scaling_factor


    def __getitem__(self, idx):
        base_file_path = os.path.join(self.root_dir, 'depth_images', self.files[idx])

        file_name = '_'.join(self.files[idx].split('_')[:-1])

        cfg_file_path = os.path.join(self.root_dir, 'cfg_files', file_name + ".cfg")
        
        profile = self._get_container_profile(cfg_file_path)



        depth_image = cv2.imread(base_file_path, cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.resize(depth_image, (self.image_size, self.image_size))
        depth_image = 1.0 - depth_image.astype(float) / 255.0


        sample = {'cross_section_profile': profile, 'depth_image': depth_image}

        if self.load_volume:
            volume_profile_path = os.path.join(self.volume_dir, file_name + ".text")
            volume_data = np.loadtxt(volume_profile_path)
            volume_profile = self._get_volume_profile(volume_data)
            sample['volume_profile'] = volume_profile

            if self.calc_wait_times:
                sample['wait_times'] = self._calc_wait_times(volume_data)

        if self.load_speed_angle_and_scale:
            sample['speed'], sample['angle'], scale = self._load_params(base_file_path)


        return sample

