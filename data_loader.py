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
    def __init__(self, root_dir, is_train, num_divisions=128, image_size=128, center=False):
        self.num_divisions = num_divisions
        self.root_dir = root_dir
        self.center = center
        self.image_size = image_size

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
                elif split[0] == 'Radius': # Hack because the flask cfgs were not correctly formatted
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


    def __getitem__(self, idx):
        base_file_path = os.path.join(self.root_dir, 'depth_images', self.files[idx])

        cfg_file_path = os.path.join(self.root_dir, 'cfg_files', '_'.join(self.files[idx].split('_')[:-1]))
        
        profile = self._get_container_profile(cfg_file_path + ".cfg")

        depth_image = cv2.imread(base_file_path, cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.resize(depth_image, (self.image_size, self.image_size))
        

        sample = {'profile': profile, 'depth_image': depth_image}

        return sample

