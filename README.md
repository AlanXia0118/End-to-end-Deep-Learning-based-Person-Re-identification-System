# CVPR2015-DL-for-Person-Re-Identification
Implementation of Deep Learning Architecture for Person Re-Identification, in CVPR2015.

# Enviroment
Packages utilized in this project including:  
* python 3.5.4  
* opencv 3.1.0  
* matplotlib 2.0.2  
* numpy 1.12.1
* keras 2.1.5

# Dataset
The current model was trained on CUHK01 dataset which consists of 971 identities, with 2 images per person in each view. You can check for more details on http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html.<br>
More specifically, we take as input cropped and augmented pictures instead of raw photos, which we will discuss later in training process.

# Overall Architecture
The `~/graph` directory contains file that can visualize the graph on Tensorboard.
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/tensorboard.png)
<br>
<br>
The design of network was mainly motivated by (cnn-bn-relu)*n structure and AlexNet.
3 dropout layers, with dropout rate all set to be 0.5, were inserted to conquer the problem of overfitting which initial model previously suffered from. This helped the model to generalize much better, as the accuracy finally raised by about 3%. 
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/arch1.png)
