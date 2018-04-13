import random
import numpy as np
import glob
import os
import cv2


# change data_path to your directory of CUHK01 dataset
data_path = '/Users/alanxia/Documents/Education/CV/Projects/ReID/CUHK01/campus'
if len(data_path) == 0:
    raise NameError('Please input your data path!')

H_5percent = 8
W_5percent = 3

# divide 971 identities
number_train_id = 871
number_test_id = 100


def translate_and_crop(img_list, img_raw):
    '''
    given one image, operate translations and crops
    append the 5 images translated, cropped and resized as well as 1 raw image to the list
    '''

    img_list.append(img_raw)

    # generate list of #pixels of vertical translation
    # and make sure there is no identical translation
    h_translation_range = np.arange(-H_5percent, H_5percent + 1).tolist()
    # delete 0 in translation range
    del h_translation_range[H_5percent]
    h_translation = random.sample(h_translation_range, 5)

    # generate list of #pixels of horizontal translation
    w_translation_range = np.arange(-W_5percent, W_5percent + 1).tolist()
    # delete 0 in translation range
    del w_translation_range[W_5percent]
    w_translation = random.sample(w_translation_range, 5)

    for i in range(0, 5):
        img_crop = img_raw[max(0, h_translation[i]): min(160, 160 + h_translation[i]),
                   max(0, w_translation[i]): min(60, 60 + w_translation[i]), :]
        img_crop = cv2.resize(img_crop, (60, 160), interpolation=cv2.INTER_CUBIC)
        img_list.append(img_crop)


def label_positive(img_channel1, img_channel2, labels, paths):
    '''
    output labels and 60 labeled image pairs for every id
    accordingly append them to lists: img_channel1, img_channel2, labels
    overall size = 36 * 871 = 31356
    '''

    # record original channel size
    original_size = len(img_channel1)
    id_num = int(len(paths) / 4)

    # create training id pairs
    for i in range(0, id_num):
        # load every picture of current identity
        j = 4 * i
        now_id1 = cv2.imread(paths[j])
        now_id2 = cv2.imread(paths[j + 1])
        now_id3 = cv2.imread(paths[j + 2])
        now_id4 = cv2.imread(paths[j + 3])
        id_list = [now_id1, now_id2, now_id3, now_id4]
        channel_list = [img_channel1, img_channel2]
        # output for different views
        for ii in range(0, 2):
            for jj in range(2, 4):
                translate_and_crop(channel_list[ii], id_list[ii])
                translate_and_crop(channel_list[1 - ii], id_list[jj])
        # output for the same view once
        translate_and_crop(img_channel1, now_id1)
        translate_and_crop(img_channel2, now_id2)
        translate_and_crop(img_channel1, now_id3)
        translate_and_crop(img_channel2, now_id4)

    # create labels
    labels_size = len(img_channel1) - original_size
    labels.append(np.tile([1, 0], (labels_size, 1)))


def label_pair_negative(img_channel1, img_channel2, id1, id2, paths):
    '''
    given one pair of id, append 2 pairs of negative training data to img_channel1 and img_channel2
    '''

    id1_view1 = cv2.imread(paths[int(4 * id1) + random.randint(0, 1)])
    id1_view2 = cv2.imread(paths[int(4 * id1) + random.randint(2, 3)])
    id2_view1 = cv2.imread(paths[int(4 * id2) + random.randint(0, 1)])
    id2_view2 = cv2.imread(paths[int(4 * id2) + random.randint(2, 3)])
    id_list = [id1_view1, id2_view1, id1_view2, id2_view2]
    for i in range(2):
        img_channel1.append(id_list[i])
        img_channel2.append(id_list[3 - i])


def label_negative(img_channel1, img_channel2, labels, paths):
    '''
    Training:
    since 871 = 67 * 13, randomly divide 871 ids into gropus of size 67
    make 2 negative examples for each pair within a certain group
    overall size = 67 * 66 / 2 * 4 * 13 = 57486
    Test:
    two groups of size 50, overall size = 4900 ~ 1.5 * overall size of positive test data
    '''

    # record original channel size
    original_size = len(img_channel1)
    id_num = int(len(paths) / 4)

    # calculate group size
    group_size = 67
    if id_num == number_test_id:
        group_size = 50
    group_num = int(id_num / group_size)

    # generate shuffle mask
    shuffle_mask = []
    mask = np.arange(id_num)
    # mask is a map from [0, ..., id_num - 1]
    np.random.shuffle(mask)
    for i in range(group_num):
        shuffle_mask.append(mask[i * group_size: (i + 1) * group_size])

    # label each group
    for i in range(group_num):
        for ii in range(group_size):
            for jj in range(ii + 1, group_size):
                label_pair_negative(img_channel1=img_channel1,
                                    img_channel2=img_channel2,
                                    id1=shuffle_mask[i][ii],
                                    id2=shuffle_mask[i][jj],
                                    paths=paths)

    # create labels
    labels_size = len(img_channel1) - original_size
    labels.append(np.tile([0, 1], (labels_size, 1)))


def main():
    # generate a list of paths of images
    path_list = glob.glob(os.path.join(data_path, '*.png'))
    path_list.sort()

    # divide the data for labelling
    train_path = path_list[0: number_train_id * 4]
    test_path = path_list[number_train_id * 4:number_train_id * 4 + number_test_id * 4]

    # label training and validation data
    img_channel1_train = []
    img_channel2_train = []
    labels_train = []
    label_positive(img_channel1=img_channel1_train,
                   img_channel2=img_channel2_train,
                   labels=labels_train,
                   paths=train_path)
    label_negative(img_channel1=img_channel1_train,
                   img_channel2=img_channel2_train,
                   labels=labels_train,
                   paths=train_path)

    # save training data
    labels_train = labels_train[0].tolist() + labels_train[1].tolist()
    labels_train = np.array(labels_train)
    labels_train.reshape(len(img_channel1_train), 2)
    print('Channel1 training examples: ', len(img_channel1_train))
    print('Channel2 training examples: ', len(img_channel2_train))
    print('Labels: ', labels_train.shape)
    np.save('channel1_train.npy', img_channel1_train)
    np.save('channel2_train.npy', img_channel2_train)
    np.save('labels_train.npy', labels_train)

    # label test data
    img_channel1_test = []
    img_channel2_test = []
    labels_test = []
    label_positive(img_channel1=img_channel1_test,
                   img_channel2=img_channel2_test,
                   labels=labels_test,
                   paths=test_path)
    label_negative(img_channel1=img_channel1_test,
                   img_channel2=img_channel2_test,
                   labels=labels_test,
                   paths=test_path)

    # save test data
    labels_test = labels_test[0].tolist() + labels_test[1].tolist()
    labels_test = np.array(labels_test)
    labels_test.reshape(len(img_channel1_test), 2)
    print('Channel1 test examples: ', len(img_channel1_test))
    print('Channel2 test examples: ', len(img_channel2_test))
    print('Labels: ', labels_test.shape)
    np.save('channel1_test.npy', img_channel1_test)
    np.save('channel2_test.npy', img_channel2_test)
    np.save('labels_test.npy', labels_test)
    print('\n----Finish---\n')


if __name__ == '__main__':
    main()