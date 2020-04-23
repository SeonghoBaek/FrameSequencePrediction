import math
import numpy as np
import tensorflow as tf
import errno
import os
import cv2


def get_batch(X, X_, size):
    # X, X_ must be nd-array
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


def get_sequence_batch(X, seq_length, batch_size):
    # print('input dim:', len(X[0]), ', seq len:', seq_length, ', batch_size:', batch_size)
    # X must be nd-array
    a = np.random.choice(len(X)-seq_length, batch_size, replace=False)
    a = a + seq_length

    # print('index: ', a)

    seq = []

    for i in range(batch_size):
        if a[i] < seq_length - 1:
            s = np.random.normal(0.0, 0.1, [seq_length, len(X[0])])
            seq.append(s)
        else:
            s = np.arange(a[i]-seq_length, a[i])
            seq.append(X[s])

    seq = np.array(seq)

    return X[a], seq


def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass


def sigmoid_normalize(value_list):
    list_max = float(max(value_list))
    alist = [i/list_max for i in value_list]
    alist = [1/(1+math.exp(-i)) for i in alist]

    return alist


def swish(logit,  name=None):
    with tf.name_scope(name):
        l = tf.multiply(logit, tf.nn.sigmoid(logit))

        return l


def generate_samples(dim, num_inlier, num_outlier, normalize=True):
    inlier = np.random.normal(0.0, 1.0, [num_inlier, dim])

    sample_inlier = []

    if normalize:
        inlier = np.transpose(inlier)

        for values in inlier:
            values = sigmoid_normalize(values)
            sample_inlier.append(values)

        inlier = np.array(sample_inlier).transpose()

    outlier = np.random.normal(1.0, 1.0, [num_outlier, dim])

    sample_outlier = []

    if normalize:
        outlier = np.transpose(outlier)

        for values in outlier:
            values = sigmoid_normalize(values)
            sample_outlier.append(values)

        outlier = np.array(sample_outlier).transpose()

    return inlier, outlier


def add_gaussian_pixel_noise(image, mean=0.0, var=0.1):
    row, col = image.shape
    #print('Add gaussian noise: ', image.shape)
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def add_gaussian_noise(input_layer, mean, std):
    if std < 0.0:
        return input_layer

    noise = tf.random_normal(shape=tf.shape(input_layer), mean=mean, stddev=std, dtype=tf.float32)
    return input_layer + noise


def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def open_capture_dev(file_path):
    try:
        vidcap = cv2.VideoCapture(file_path)
        success, _ = vidcap.read()

        if success is False:
            print('File Reading Failed. ' + file_path)
            return None
    except:
        print('Error Opening Video Capture Device or File' + file_path)
        return None

    return vidcap


def get_frame_sequece(c_dev, sequence_length, skip=4, b_crop=False, crop_box=[0, 0, 0, 0]):
    sequence = 1
    count = 0
    target_length = sequence_length

    if sequence_length < 0:
        target_length = 1
    
    while True:
        success, img = c_dev.read()

        if success is True:
            count += 1

            if count % skip == 0:
                if sequence_length > 0 and count > sequence_length:
                    break
                if b_crop is True:
                    img = img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
                cv2.imwrite(str(sequence) + '.jpg', img)
                sequence += 1
        else:
            print('File Reading Failed.')
            return None

    return sequence


def read_stream(dev_file, crop_box=[56, 1080, 0, 1920]):
    cap_dev = open_capture_dev(dev_file)

    if cap_dev is not None:
        #images = get_frame_sequece(cap_dev, -1, skip=15, b_crop=True, crop_box=[140, 840, 820, 1520])
        images = get_frame_sequece(cap_dev, -1, skip=8, b_crop=True, crop_box=[56, 1080, 0, 1920])
        print(len(images))


def get_image_gradients(in_image):
    # in_image = numpy array
    h, w, c = in_image.shape

    # h dim
    a = in_image[1:, :, :]
    b = in_image[:-1, :, :]
    grad_h = a - b

    # w dim
    a = in_image[:, 1:, :]
    b = in_image[:, :-1, :]
    grad_w = a - b

    return grad_h, grad_w

# read_stream('/train.mp4')

