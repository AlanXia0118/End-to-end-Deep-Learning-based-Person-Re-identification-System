from keras.models import Model
from model import generate_model
import numpy as np
import h5py

# load model and mean_img
model = generate_model()
model.load_weights('./model/reid_model.h5')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
mean_img1 = np.load('./mean_img/mean_img1.npy')
mean_img2 = np.load('./mean_img/mean_img2.npy')

# load and pre-process data
x_test1 = np.load('channel1_test.npy') / 255
x_test2 = np.load('channel2_test.npy') / 255
y_test = np.load('labels_test.npy')
x_test1 -= mean_img1
x_test2 -= mean_img2

# test on data
loss, accuracy = model.evaluate([x_test1, x_test2], y_test, batch_size=128)
print('Test set:')
print('loss = {0}, accuracy = {1}.'.format(loss, accuracy))
