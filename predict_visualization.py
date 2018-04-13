from keras.models import Model
from model import generate_model
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

# predict on your own pairs of identities
img_path1 = './test_dataset/4_1.png'
img_path2 = './test_dataset/8_1.png'


# load model and mean image of training data
model = generate_model()
model.load_weights('./model/reid_model.h5')
mean_img1 = np.load('./mean_img/mean_img1.npy')
mean_img2 = np.load('./mean_img/mean_img2.npy')


# pre-processing the given pair of images
img1_raw = cv2.imread(img_path1)
img2_raw = cv2.imread(img_path2)
size = (60, 160)
img1_raw = cv2.resize(img1_raw, size, interpolation=cv2.INTER_AREA)
img2_raw = cv2.resize(img2_raw, size, interpolation=cv2.INTER_AREA)
img1 = np.array([img1_raw/255 - mean_img1])
img2 = np.array([img2_raw/255 - mean_img2])


# predict for images
result = ['same', 'diff']
prediction = model.predict([img1, img2])
print('The probability of being same is :')
print(prediction[0][0])
print('The probability of being different is :')
print(prediction[0][1])


# prediction visualization
img_show = np.zeros(shape=(160, 120, 3))
img_show[0:160, 0:60] += img1_raw[:, :]
img_show[0:160, 60:120] += img2_raw[:, :]
img_show = img_show[:, :, (2, 1, 0)].astype(np.uint8)
plt.ion()
plt.imshow(img_show)
plt.text(20, 80, result[int(np.argmax(prediction[0]))], size=60, alpha=1, color='w')
plt.pause(5)
