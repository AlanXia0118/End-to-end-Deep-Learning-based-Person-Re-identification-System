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
More specifically, we take as input cropped and augmented pictures instead of raw photos, which will be discussed later in training process.

# Overall Architecture
As represented explicitly in the paper, the model is composed of tied convolution layers and deeper layers that compute relationships between two pictures, sharing similar attributes with Siamese neural networks.<br>
You can probably gain more intution by referring to related work, or check out `model.py`.
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/DL-for-ReID/model.png)
<br>
<br>

# Visualized prediction
`predict_visualization.py` is prepared for predicting on your own pairs of identities, employing opencv and matplotlib packages to realize the visualization. You can change the paths to be your model and image at the start of the script:

```
# predict on your own pairs of identities
img_path1 = './test_dataset/9_1.png'
img_path2 = './test_dataset/8_1.png'
```
A pre-trained model was kept for you in `~/model` which you'll have to unzip first. Since we trained the model using normalized data, mean images of dataset are required for predicting. Accordingly, one mean image for each channel is provided in `~/mean_img`. You should see a visualized prediction as below:
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/DL-for-ReID/pre_same.png)
<br>
<br>
Feel free to predict on pictures we collected and stored in `~/test_dataset`.



# Training and validation
The training process is executed in `model_train_and_val.py` as well as validation. Since the model is overall end-to-end, you can start training by feeding each channel of inputs to the first layer just like any other typical CNN architecture.

However, according to the paper, we firstly implement data augmentation and hard negative mining in `npydata_generator.py`, which produces ndarray datasets including training data and test data catering to Keras architectures, to acquire sufficient data and address the problem of imbalanced dataset. 

* Data Augmentation<br>
From 971 original identities, with 2 images per person in each view in CUHK01, due to the protocol that it is better suited for deep learning to use 90% of the data for training, we divided the dataset and took 871 identities for training process. By defining and calling method `translate_and_crop()`, we sample 5 images around the image center with given translation range`[-0.05H, 0.05H]×[-0.05W, 0.05W]`, which was depicted exactly in the paper. Then we utilize `label_positive()` to label a batch of augmented(i.e. newly sampled) picture pairs postive. There is actullay flexibility in how to feed pairs of images to 2 different channels, and here the strategy we employ is to ensure symmetry so that each channel should be robust to the uncertainty of input view. The final size of positive pairs should be 31356, as detailed computation was shown in the `npydata_generator.py`.
```
  label_positive(img_channel1=img_channel1_train,
                 img_channel2=img_channel2_train,
                 labels=labels_train,
                 paths=train_path)
```

* Hard Negative Mining<br>
Data augmentation increases the number of positive pairs, but imbalance in dataset may still be possibly invited by directly generating negative pairs from all raw images. Therefor, we randomly group the dataset(e.g. divide 871 identities into 13 groups of 67 identities) and label each pair inside negative by calling method `label_negative()`, and manage to maintain the overall size of negative examples twice as many as postive examples as emphasized in the referrence paper.
```
  label_negative(img_channel1=img_channel1_train,
                   img_channel2=img_channel2_train,
                   labels=labels_train,
                   paths=train_path)
```

In the `model_train_and_val.py` code, we firstly sort of normalize and shuffle the ndarray dataset and reserve 20% of them for validation. Due to several randomly sampling operations, each time running the `npydata_generator.py` should produce different datasets, so we actually use different parts of training data to train and evaluate the model, which share the similar idea of Cross-validation. You shoul see an evaluation result as below:
```
Validation set:
loss = 0.10798121768765487, accuracy = 0.9823287748325736.
```
And the model will be automatically saved to the root directory, named as `reid_model.h5`.

# Test the model
Similarly, you can test the trained model by running the following command and get the result formatted as below:
```
python model_test.py


Test set:
loss = 0.15826140906530267, accuracy = 0.9650588235294117.
```

# Contact
For further discussion, you are welcome to contact me at ```kedi_xia@zju.edu.cn```.
