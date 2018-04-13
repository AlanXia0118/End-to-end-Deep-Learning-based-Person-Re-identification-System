from model import generate_model
from keras import optimizers
from keras import  models
import numpy as np
import h5py
import tensorflow as tf
import cv2


def main():

    # pre-processing data
    x_train1 = np.load('channel1_train.npy')/255
    x_train2 = np.load('channel2_train.npy')/255
    y_train = np.load('labels_train.npy')
    mean_img1 = np.mean(x_train1, axis=0)
    mean_img2 = np.mean(x_train2, axis=0)
    np.save('mean_img1.npy', mean_img1)
    np.save('mean_img2.npy', mean_img2)
    x_train1 -= mean_img1
    x_train2 -= mean_img2

    # shuffle with a mask
    shuffle_mask = np.arange(0, x_train1.shape[0])
    np.random.shuffle(shuffle_mask)
    x_train1 = x_train1[shuffle_mask, :]
    x_train2 = x_train2[shuffle_mask, :]
    y_train = y_train[shuffle_mask, :]

    # divide data into 2 groups: training and validation
    train_size = int(x_train1.shape[0]*0.8)
    x_training1 = x_train1[0: train_size, :]
    x_training2 = x_train2[0: train_size, :]
    y_training = y_train[0: train_size, :]
    x_val1 = x_train1[train_size:, ]
    x_val2 = x_train2[train_size:, ]
    y_val = y_train[train_size:, ]
    print('Training data 1 size:', x_training1.shape)
    print('Training data 2 size:', x_training2.shape)
    print('Training labels size:', y_training.shape)
    print('Evaluation data 1 size:', x_val1.shape)
    print('Evaluation data 2 size:', x_val2.shape)
    print('Evaluation labels size:', y_val.shape)


    # compile the model
    model = generate_model()
    model.summary()
    adam_optimizer = optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    # training
    model.fit([x_train1, x_train2], y_train, epochs=2, batch_size=64)
    print('\n------train_done------\n')
    model.save('reid_model.h5')


    # evaluate validation set
    loss, accuracy = model.evaluate([x_val1, x_val2], y_val)
    print('Validation set:')
    print('loss = {0}, accuracy = {1}.'.format(loss, accuracy))

    # finish
    print('\n----Finish----\n')


if __name__ == '__main__':
    main()




