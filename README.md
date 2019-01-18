# nn_container_shape_from_depth


This repository implements the training for a network used to predict the amount of time it takes for liquid to exit a container given the container's cross sectional profile.  This work is described in the following paper.


[Autonomous Precision Pouring from Unknown Symmetric Containers](https://www.seas.upenn.edu/~kmonroe/Autonomous_Precision_Pouring_from_Unknown_Containers.pdf)

Monroe Kennedy III,  Karl Schmeckpeper, Dinesh Thakur, Chenfanfu Jiang, Vijay Kumar and Kostas Daniilidis

Preprint 2018.

In addition to the functionality described in the paper, this repository also supports the training of networks to predict the theta vs volume profile from a depth image of the container or the container’s cross section, networks to predict the time it takes for a liquid to exit a container given a depth image or a cross section, and networks to predict the cross sectional profile from a depth image.

All containers are assumed to be surfaces of revolution.


# Usage

`python train.py -h`
