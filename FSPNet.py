# Video Frame Sequence Prediction using CPC
# Author: Seongho Baek seonghobaek@gmail.com


import tensorflow as tf
import layers
from sklearn.utils import shuffle
import numpy as np
import os
import argparse
import cv2
import util
import operator


def get_global_context_area(img, area):
    # img should be numpy array
    context_area = img[area[0]:area[1], area[2]:area[3]].copy()

    return context_area


def get_image_batches(folder, start, batch_size, use_sobel=False):
    batch_start = start
    batch_end = batch_start + batch_size

    images = []
    global_images = []

    for idx in range(batch_start, batch_end):
        filename = str(idx) + '.jpg'
        fullname = os.path.join(folder, filename).replace("\\", "/")
        #print('image file name: ' + fullname)
        jpg_img = cv2.imread(fullname)
        jpg_img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2GRAY)  # To RGB format
        img = cv2.resize(jpg_img, dsize=(input_width, input_height))

        '''
        img = np.reshape(img, -1)
        img = [x * 0.7 if x > 224.0 else x for x in img]
        img = np.reshape(img, (input_height, input_width))
        '''

        if use_sobel is True:
            img_so_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            img_so_x = cv2.convertScaleAbs(img_so_x)
            img_so_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            img_so_y = cv2.convertScaleAbs(img_so_y)
            img = cv2.addWeighted(img_so_x, 1, img_so_y, 1, 0)
            #cv2.imwrite('batch_' + str(idx) + '.jpg', img)

        if img is not None:
            img = np.array(img)
            #n_img = img / 255.0
            n_img = img
            images.append(n_img)

        #g_img = get_global_context_area(jpg_img, crop_global_area)
        #g_img = cv2.resize(g_img, dsize=(global_area_width, global_area_height))
        #global_images.append(g_img)
    if len(images) < batch_size:
        print('Batch creation failed.')

    return np.array(images)


def load_images_from_folder(folder, use_augmentation=False, add_noise=False):
    images = []

    # To Do
    # Color, Brightness Augmentation

    sorted_list = os.listdir(folder)
    sorted_list.sort()

    for idx in range(len(sorted_list)):
        filename = str(idx+1) + '.jpg'
        fullname = os.path.join(folder, filename).replace("\\", "/")
        jpg_img = cv2.imread(fullname)
        #img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)  # To RGB format
        img = cv2.resize(jpg_img, dsize=(input_width, input_height))

        if img is not None:
            img = np.array(img)
            n_img = img / 255.0
            mean = np.mean(n_img)
            std = np.std(n_img)
            n_img = (n_img - mean) / std
            images.append(n_img)

    return np.array(images)


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    loss = gamma * loss

    return loss


def get_diff_loss(anchor, positive, negative):
    a_p = get_residual_loss(anchor, positive, 'l2')
    a_n = get_residual_loss(anchor, negative, 'l2')
    # a_n > a_p + margin
    # a_p - a_n + margin < 0
    # minimize (a_p - a_n + margin)
    return a_p / a_n


def get_gradient_loss(img1, img2):
    image_a = tf.expand_dims(img1, axis=0)
    image_b = tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    v_a = tf.reduce_mean(tf.image.total_variation(image_a))
    v_b = tf.reduce_mean(tf.image.total_variation(image_b))

    #loss = tf.abs(tf.subtract(v_a, v_b))
    loss = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b))) + tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

    return loss


def auto_regressive(latents, sequence_length=4, skip=0, out_dim=256, scope='ar'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # [Batch Size, Latent Dims]
        sequence_batch = tf.stack([tf.slice(latents, [i, 0], [sequence_length, -1]) for i in range(batch_size - sequence_length - skip)], axis=0)
        #sequence_batch = tf.slice(latents, [0, 0], [sequence_length-skip, -1])
        #sequence_batch = tf.expand_dims(sequence_batch, 0)
        print('Sequence Batch Shape: ' + str(sequence_batch.get_shape().as_list()) + ', skip: ' + str(skip))

        context = layers.bi_lstm_network(sequence_batch, lstm_hidden_size_layer=ar_lstm_hidden_layer_dims, lstm_latent_dim=out_dim)
        #context = layers.lstm_network(sequence_batch, lstm_hidden_size_layer=ar_lstm_hidden_layer_dims, lstm_latent_dim=out_dim)

    return context


def CPC(latents, target_dim=64, emb_scale=0.1, scope='cpc'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        context = auto_regressive(latents, sequence_length=ar_lstm_sequence_length, skip=predict_skip, out_dim=ar_context_dim)
        # [num_predict, ar_context_dim]
        print('AR Context Shape: ' + str(context.get_shape().as_list()))

        # One Hot Label
        onehot_labels = []

        for i in range(batch_size - ar_lstm_sequence_length - predict_skip):
            target_index = i + ar_lstm_sequence_length + predict_skip
            onehot = np.zeros(batch_size)
            onehot[target_index] = 1
            print('Label: ' + str(onehot))
            onehot_labels.append(onehot)

        onehot_labels = np.array(onehot_labels)
        onehot_labels = tf.constant(onehot_labels)

        num_predicts = batch_size - ar_lstm_sequence_length - predict_skip

        def context_transform(input_layer, scope='context_transform'):
            fc1 = layers.fc(input_layer, ar_context_dim//2, use_bias=True, non_linear_fn=tf.nn.relu, scope=scope + '_fc1')
            fc2 = layers.fc(fc1, target_dim, use_bias=True, non_linear_fn=None, scope=scope + '_fc2')

            return fc2

        #scaled_context = tf.stack([layers.fc(context, target_dim, use_bias=True, non_linear_fn=tf.nn.sigmoid, scope='pred_' + str(i)) for i in range(num_predicts)])
        scaled_context = context_transform(context, scope='pred')
        #scaled_context = tf.squeeze(scaled_context)

        print('Predict Shape: ' + str(scaled_context.get_shape().as_list()))

        scaled_context = scaled_context * emb_scale
        targets = latents
        #targets = layers.fc(targets, target_dim, use_bias=True, non_linear_fn=tf.nn.sigmoid, scope='target')
        #targets = tf.slice(latents, [ar_lstm_sequence_length + predict_skip, 0], [num_predicts, -1])
        #targets = context_transform(targets, scope='target')
        targets = context_transform(targets, scope='targets')
        print('Target Shape: ' + str(targets.get_shape().as_list()))

        #scaled_context = tf.expand_dims(scaled_context, 0)
        logits = tf.matmul(scaled_context, targets, transpose_b=True)
        print('Logit Shape: ' + str(logits.get_shape().as_list()))

        entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

        return entropy_loss, logits, context[-1]


def decoder(latent, anchor_layer=None, activation='swish', scope='decoder_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = dense_block_depth * 16

        l = latent
        #l = layers.fc(l, 4*4*32, non_linear_fn=act_func)

        print('Decoder Input: ' + str(latent.get_shape().as_list()))
        transform_channel = latent.get_shape().as_list()[-1] // 16

        l = tf.reshape(l, shape=[-1, 4, 4, transform_channel])

        l = layers.conv(l, scope='tr0', filter_dims=[1, 1, block_depth], stride_dims=[1, 1], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln0')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn0')

        l = act_func(l)

        for i in range(10):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                   act_func=act_func, norm=norm, b_train=b_train, scope='block0_' + str(i))

        print('Decoder Block 0: ' + str(l.get_shape().as_list()))

        if anchor_layer is not None:
            block_depth = dense_block_depth * 8
            # 8 x 8
            l = layers.deconv(l, b_size=1, scope='deconv1', filter_dims=[3, 3, block_depth],
                              stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

            print('Decoder Anchor: ' + str(anchor_layer.get_shape().as_list()))
            #anchor_layer = tf.stop_gradient(anchor_layer)
            l = tf.concat([l, anchor_layer], axis=3)

            block_depth = block_depth * 2
        else:
            block_depth = dense_block_depth * 16
            # 8 x 8
            l = layers.deconv(l, b_size=1, scope='deconv1', filter_dims=[3, 3, block_depth],
                              stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

        print('Deconvolution 1: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                   act_func=act_func, norm=norm, b_train=b_train, scope='block1_' + str(i))

        # 16 x 16
        block_depth = dense_block_depth * 8

        l = layers.deconv(l, b_size=1, scope='deconv2', filter_dims=[3, 3, block_depth],
                             stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

        print('Deconvolution 2: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                   act_func=act_func, norm=norm, b_train=b_train, scope='block2_' + str(i))

        # 32 x 32
        block_depth = dense_block_depth * 2

        l = layers.deconv(l, b_size=1, scope='deconv3', filter_dims=[3, 3, block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

        print('Deconvolution 3: ' + str(l.get_shape().as_list()))

        '''
        if anchor_layer is not None:

            # 32 x 32
            print('Decoder Anchor: ' + str(anchor_layer.get_shape().as_list()))
            #anchor_layer = tf.stop_gradient(anchor_layer)
            l = tf.concat([l, anchor_layer], axis=3)

            block_depth = block_depth * 2
        '''
        for i in range(10):
            #l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
            #                          act_func=act_func, norm=norm, b_train=b_train, scope='block3_' + str(i))
            l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                act_func=act_func, norm=norm, b_train=b_train,
                                                scope='dense_block_1_' + str(i))

        l = layers.conv(l, scope='tr1', filter_dims=[1, 1, num_channel], stride_dims=[1, 1], non_linear_fn=act_func)

        print('Decoder Final: ' + str(l.get_shape().as_list()))

        return l


def encoder(x, activation='relu', scope='encoder_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Encoder Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth * 2

        l = x
        l = layers.conv(l, scope='conv1', filter_dims=[7, 7, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn')

        l = act_func(l)

        for i in range(4):
            l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                         act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1_' + str(i))

        #l = layers.self_attention(l, block_depth)

        #anchor_layer = tf.slice(l, [l.get_shape().as_list()[0] - predict_skip - 2, 0, 0, 0], [1, -1, -1, -1])
        block_depth = block_depth * 2

        # [16 x 16]
        l = layers.conv(l, scope='tr1', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln1')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn1')

        l = act_func(l)

        print('Encoder Block 0: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_1_' + str(i))

        block_depth = block_depth * 2

        # [8 x 8]
        l = layers.conv(l, scope='tr2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln2')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn2')

        l = act_func(l)

        print('Encoder Block 1: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_2_' + str(i))

        anchor_layer = tf.slice(l, [l.get_shape().as_list()[0] - 2 - predict_skip, 0, 0, 0], [1, -1, -1, -1])
        print('Anchor Layer: ' + str(anchor_layer.get_shape().as_list()))

        block_depth = block_depth * 2

        # [4 x 4]
        l = layers.conv(l, scope='tr3', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        print('Encoder Block 2: ' + str(l.get_shape().as_list()))

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln3')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn3')

        l = act_func(l)

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_3_' + str(i))

        l = layers.conv(l, scope='tr4', filter_dims=[1, 1, representation_dim], stride_dims=[1, 1], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln4')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn4')

        l = act_func(l)

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, representation_dim], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_4_' + str(i))

        last_layer = l

        context = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=True, scope='gp')
        print('Encoder GP Dims: ' + str(context.get_shape().as_list()))

    return context, anchor_layer


def global_encoder(x, activation='relu', scope='global_encoder_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Global Encoder Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        l = x
        l = layers.conv(l, scope='conv1', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)

        #l = layers.self_attention(l, block_depth)

        for i in range(3):
            l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                         act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1_' + str(i))

        # l = layers.self_attention(l, block_depth)

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr1')
        # [60, 80]
        print('Global Encoder Block 0: ' + str(l.get_shape().as_list()))

        for i in range(4):
            l = layers.add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=4, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_1_' + str(i), use_bottleneck=True)

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr2')

        # [30, 40]
        print('Global Encoder Block 1: ' + str(l.get_shape().as_list()))

        for i in range(4):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_3_' + str(i), use_bottleneck=True)

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr4')

        # [15, 20]
        print('Global Encoder Block 2: ' + str(l.get_shape().as_list()))

        for i in range(4):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_4_' + str(i), use_bottleneck=True)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr5')

        # [7, 10]
        print('Global Encoder Block 3: ' + str(l.get_shape().as_list()))

        for i in range(4):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_5_' + str(i), use_bottleneck=True)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim], stride_dims=[1, 1],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr6')

        last_layer = act_func(l)

        context = layers.global_avg_pool(last_layer, output_length=representation_dim//2, use_bias=True, scope='gp')
        print('Global Encoder GP Dims: ' + str(context.get_shape().as_list()))

    return context


def check_patch_changeness(patch_list, patch_index, validity=0, threshold=0.0, train_mode=False):
    num_patches = len(patch_list)

    #safe_region = 1 + num_patches // 2  # 50 % front area
    safe_region = num_patches - 2 - predict_skip  # Index start from 0

    # Test time only
    if train_mode is False:
        current_patch = patch_list[-1] # target patch
        next_patch = patch_pixel_mean[patch_index]
        changeness = np.abs(current_patch - next_patch)
        m = np.mean(np.array(changeness), axis=(0, 1))

        if m < threshold:
            # print('Frame ' + str(i) + ' Patch ' + str(patch_index) + ' Changeness: ' + str(m))
            return 0

    # Fast traverse
    if validity == 1:
        for i in range(safe_region, -1, -1):
            current_patch = patch_list[i]
            next_patch = patch_pixel_mean[patch_index]
            changeness = np.abs(current_patch - next_patch)
            m = np.mean(np.array(changeness), axis=(0, 1))

            if m > threshold:
                # print('Frame ' + str(i) + ' Patch ' + str(patch_index) + ' Changeness: ' + str(m))
                return 1
    else:
        current_patch = patch_list[safe_region]
        next_patch = patch_pixel_mean[patch_index]
        changeness = np.abs(current_patch - next_patch)
        m = np.mean(np.array(changeness), axis=(0, 1))

        if m > threshold:
            # print('Frame ' + str(safe_region) + ' Patch ' + str(patch_index) + ' Changeness: ' + str(m))
            return 1

    return 0


def prepare_patches(image, patch_size=[24, 24], patch_dim=[7, 7], stride=12, add_noise=True):
    patches = []
    patch_w = patch_size[0]
    patch_h = patch_size[1]

    # print('image shape: ', image.shape)
    for h in range(patch_dim[0]):
        for w in range(patch_dim[1]):
            # print('h:', h*stride, ' w: ', h*stride + patch_h)
            # print('w:', w*stride, ' w: ', w*stride + patch_w)
            patch = image[h*stride:(h*stride + patch_h), w*stride:(w*stride + patch_w)].copy()
            if add_noise is True:
                patch = util.add_gaussian_pixel_noise(patch, mean=0.0, var=25.0)
            patches.append(patch)

    # print('Num patches: ', len(patches))
    return np.array(patches)


def fine_tune(base_model_path, task_model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])
        Z = tf.placeholder(tf.float32, [batch_size, global_area_height, global_area_width, num_channel])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latents, anchor_layer = encoder(X, activation='swish', norm='layer', b_train=b_train, scope='encoder')
    # [Batch Size, Latent Dims]
    print('Encoder Dims: ' + str(latents.get_shape().as_list()))

    # latents_global = global_encoder(Z, activation='swish', norm='layer', b_train=b_train, scope='g_encoder')
    # print('Global Encoder Dims: ' + str(latents_global.get_shape().as_list()))

    # latents = tf.concat([latents, latents_global], axis=-1)

    print('Final Encoder Dims: ' + str(latents.get_shape().as_list()))

    cpc_e_loss, cpc_logits, cpc_context = CPC(latents, target_dim=cpc_target_dim, emb_scale=1.0, scope='cpc')

    reconstructed_patch = decoder(cpc_context, anchor_layer=anchor_layer, activation='swish', norm='layer',
                                  b_train=b_train, scope='decoder')

    reconstructed_patch = tf.squeeze(reconstructed_patch, axis=[0])

    print('Reconstructed Patch Dims: ' + str(reconstructed_patch.get_shape().as_list()))

    r_loss = get_residual_loss(reconstructed_patch, Y, type='l1')
    grad_loss = get_gradient_loss(reconstructed_patch, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(cpc_e_loss + r_loss + grad_loss)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, task_model_path)
            print('Model Restored')
        except:
            try:
                variables_to_restore = [v for v in tf.trainable_variables()
                                        if v.name.split('/')[0] == 'encoder'
                                        or v.name.split('/')[0] == 'cpc'
                                        or v.name.split('/')[0] == 'decoder']

                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, base_model_path)

                print('Partial Model Restored. Start fine tune. Wait ...')
            except:
                print('Start New Training. Wait ...')

        trX = os.listdir(train_data)
        print('Number of Training Images: ' + str(len(trX)))
        total_num_batches = len(trX) - batch_size

        for e in range(num_epoch):
            itr_save = 0
            prev_patch_batch = []
            valid_status = np.zeros(num_context_patches_width * num_context_patches_height)

            for start in range(total_num_batches):
                if len(prev_patch_batch) > 0:
                    prev_patch_batch.pop(0)

                set_size = batch_size - len(prev_patch_batch)
                if start > 0:
                    img_batches = get_image_batches(train_data, start + batch_size, set_size)
                else:
                    img_batches = get_image_batches(train_data, start, set_size)

                # img_globals = np.expand_dims(img_globals, axis=-1)

                # Create patches. batch_size * num_context_patches * num_context_patches * channel
                for i in range(set_size):
                    patches = prepare_patches(img_batches[i], patch_size=[patch_height, patch_width],
                                              patch_dim=[num_context_patches_height, num_context_patches_width],
                                              stride=patch_height // 2)
                    prev_patch_batch.append(patches)

                patch_batch = np.array(prev_patch_batch)
                patch_batch = np.stack(patch_batch, axis=1)
                patch_batch = np.expand_dims(patch_batch, axis=-1)

                loss_list = []
                g_loss_list = []
                c_loss_list = []
                index_list = []

                for i in in_list:
                    # Check validity
                    b_valid = check_patch_changeness(patch_batch[i], i, valid_status[i],
                                                     threshold=patch_changeness_threshold)
                    valid_status[i] = b_valid

                    if b_valid == 0:
                        continue

                    _, residual_loss, gradient_loss, cpc_loss = sess.run([optimizer, r_loss, grad_loss, cpc_e_loss],
                                                                         feed_dict={X: patch_batch[i],
                                                                                    Y: patch_batch[i][-1],
                                                                                    b_train: True})
                    loss_list.append(residual_loss)
                    g_loss_list.append(gradient_loss)
                    c_loss_list.append(cpc_loss)
                    index_list.append(i)

                    if residual_loss > 4000:
                        r_patch = sess.run([reconstructed_patch], feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: True})
                        cv2.imwrite(str(start+batch_size) + '_patch_' + str(i) + '.' + str(residual_loss) + '.jpg', patch_batch[i][-1])
                        cv2.imwrite(str(start+batch_size) + '_r_patch_' + str(i) + '.' + str(residual_loss) + '.jpg', r_patch[0])

                if len(loss_list) > 0:
                    max_index = loss_list.index(max(loss_list))
                    patch_index = index_list[max_index]

                    print('Epoch: ' + str(e) + ', ' + str(start + 1) + '/' + str(total_num_batches) +
                          ', Target Frame: ' + str(start + batch_size) +
                          ', Trained: ' + str(index_list))
                    print('       Worst #' + str(patch_index) + ': ' +
                          ' residual ' + str(loss_list[max_index]) +
                          ', gradient ' + str(g_loss_list[max_index]) +
                          ', cpc ' + str(c_loss_list[max_index]))
                else:
                    print('Epoch: ' + str(e) + ', ' + str(start + 1) + '/' + str(total_num_batches) +
                          ', Target Frame: ' + str(start + batch_size))

                itr_save += 1

                if itr_save % 100 == 0:
                    try:
                        print('Saving Model...')
                        saver = tf.train.Saver()
                        saver.save(sess, task_model_path)
                        print('Saved.')
                    except:
                        print('Save failed')


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])
        Z = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latents, anchor_layer = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    # [Batch Size, Latent Dims]
    print('Encoder Dims: ' + str(latents.get_shape().as_list()))

    #latents_global = global_encoder(Z, activation='swish', norm='layer', b_train=b_train, scope='g_encoder')
    #print('Global Encoder Dims: ' + str(latents_global.get_shape().as_list()))

    #latents = tf.concat([latents, latents_global], axis=-1)

    print('Final Encoder Dims: ' + str(latents.get_shape().as_list()))

    cpc_e_loss, cpc_logits, cpc_context = CPC(latents, target_dim=cpc_target_dim, emb_scale=1.0, scope='cpc')

    reconstructed_patch = decoder(cpc_context, anchor_layer=anchor_layer, activation='relu', norm='layer', b_train=b_train, scope='decoder')

    reconstructed_patch = tf.squeeze(reconstructed_patch, axis=[0])

    print('Reconstructed Patch Dims: ' + str(reconstructed_patch.get_shape().as_list()))

    r_loss = get_residual_loss(reconstructed_patch, Y, type='l2')
    grad_loss = get_gradient_loss(reconstructed_patch, Y)
    diff_loss = get_diff_loss(reconstructed_patch, Y, Z)

    alpha = 0.1
    beta = 0.9

    total_loss = beta * r_loss * (alpha + diff_loss) + (1 - beta) * grad_loss

    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(cpc_e_loss + total_loss + latent_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(total_loss + cpc_e_loss)
    softmax_cpc_logits = tf.nn.softmax(logits=cpc_logits)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        trX = os.listdir(train_data)
        print('Number of Training Images: ' + str(len(trX)))
        total_num_batches = len(trX) - batch_size

        for e in range(num_epoch):
            itr_save = 0
            prev_patch_batch = []
            valid_status = np.zeros(num_context_patches_width * num_context_patches_height)
            hard_set_batch_index = []
            hard_set_loss_list = []

            for start in range(total_num_batches):
                if len(prev_patch_batch) > 0:
                    prev_patch_batch.pop(0)

                set_size = batch_size - len(prev_patch_batch)
                if start > 0:
                    img_batches = get_image_batches(train_data, start + batch_size, set_size)
                else:
                    img_batches = get_image_batches(train_data, start, set_size)

                #img_globals = np.expand_dims(img_globals, axis=-1)

                # Create patches. batch_size * num_context_patches * num_context_patches * channel
                for i in range(set_size):
                    patches = prepare_patches(img_batches[i], patch_size=[patch_height, patch_width],
                                              patch_dim=[num_context_patches_height, num_context_patches_width],
                                              stride=patch_height//2)
                    prev_patch_batch.append(patches)

                patch_batch = np.array(prev_patch_batch)
                patch_batch = np.stack(patch_batch, axis=1)
                patch_batch = np.expand_dims(patch_batch, axis=-1)

                loss_list = []
                g_loss_list = []
                c_loss_list = []
                index_list = []
                diff_loss_list = []

                for i in in_list:
                    # Check validity
                    b_valid = check_patch_changeness(patch_batch[i], i, valid_status[i], threshold=patch_changeness_threshold, train_mode=True)
                    valid_status[i] = b_valid

                    if b_valid == 0:
                        continue

                    _, residual_loss, cpc_loss, d_loss = sess.run([optimizer, total_loss, cpc_e_loss, diff_loss],
                                                                  feed_dict={X: patch_batch[i],
                                                                             Y: patch_batch[i][-1],
                                                                             Z: patch_batch[i][-2 - predict_skip],
                                                                             b_train: True})
                    loss_list.append(residual_loss)
                    # g_loss_list.append(gradient_loss)
                    c_loss_list.append(cpc_loss)
                    index_list.append(i)
                    diff_loss_list.append(d_loss)

                    if (start + 1) % 30 == 0:
                        r_patch = sess.run([reconstructed_patch],
                                           feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: True})
                        cv2.imwrite(str(start + batch_size) + '_patch_' + str(i) + '_anchor.jpg',
                                    patch_batch[i][-2 - predict_skip])
                        cv2.imwrite(str(start + batch_size) + '_patch_' + str(i) + '_pred.jpg',
                                    r_patch[0])
                        cv2.imwrite(
                            str(start + batch_size) + '_patch_' + str(i) + '_target.jpg',
                            patch_batch[i][-1])

                if len(loss_list) > 0:
                    max_index = loss_list.index(max(loss_list))
                    patch_index = index_list[max_index]

                    print('Epoch: ' + str(e) + ', ' + str(start + 1) + '/' + str(total_num_batches) +
                          ', Target Frame: ' + str(start + batch_size))
                          # ', Trained: ' + str(index_list))
                    print('       Worst #' + str(patch_index) + ': ' +
                          ' residual ' + str(loss_list[max_index]) +
                          ', diff ' + str(diff_loss_list[max_index]) +
                          ', cpc ' + str(c_loss_list[max_index]))

                    # Save hard set
                    hard_set_batch_index.append(start)
                    hard_set_loss_list.append(loss_list[max_index])

                else:
                    print('Epoch: ' + str(e) + ', ' + str(start + 1) + '/' + str(total_num_batches) +
                          ', Target Frame: ' + str(start + batch_size))

                itr_save += 1

                if itr_save % 100 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')

            # Get top n max
            '''
            indexed_hard_set_loss_list = list(enumerate(hard_set_loss_list))
            top_n = len(hard_set_loss_list) // 30
            top_n = sorted(indexed_hard_set_loss_list, key=operator.itemgetter(1))[-top_n:]
            top_n = list(reversed([i for i, v in top_n]))
            hard_set_batch_index = np.array(hard_set_batch_index)
            hard_set_batch_index = list(hard_set_batch_index[top_n])
            num_training = 30

            for num_hard_training in range(num_training):
                for start in hard_set_batch_index:
                    img_batches = get_image_batches(train_data, start, batch_size)
                    patch_batch = []

                    # Create patches. batch_size * num_context_patches * num_context_patches * channel
                    for i in range(batch_size):
                        patches = prepare_patches(img_batches[i], patch_size=[patch_height, patch_width],
                                                  patch_dim=[num_context_patches_height, num_context_patches_width],
                                                  stride=patch_height // 2)
                        patch_batch.append(patches)

                    patch_batch = np.array(patch_batch)
                    patch_batch = np.stack(patch_batch, axis=1)
                    patch_batch = np.expand_dims(patch_batch, axis=-1)

                    for i in in_list:
                        b_valid = check_patch_changeness(patch_batch[i], i, 0,
                                                         threshold=patch_changeness_threshold)

                        if b_valid == 0:
                            continue

                        _, residual_loss, gradient_loss, cpc_loss = sess.run([optimizer, r_loss, grad_loss, cpc_e_loss],
                                                                             feed_dict={X: patch_batch[i],
                                                                                        Y: patch_batch[i][-1],
                                                                                        Z: patch_batch[i][-2 - predict_skip],
                                                                                        b_train: True})
                        print('Hard set train(' + str(num_hard_training + 1) + '/' + str(num_training) +
                              '). residual_loss: ' + str(residual_loss))
                try:
                    print('Saving model...')
                    saver.save(sess, model_path)
                    print('Saved.')
                except:
                    print('Save failed')
                '''


def test(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])
        Z = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latents, anchor_layer = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    # [Batch Size, Latent Dims]
    print('Encoder Dims: ' + str(latents.get_shape().as_list()))

    # latents_global = global_encoder(Z, activation='swish', norm='layer', b_train=b_train, scope='g_encoder')
    # print('Global Encoder Dims: ' + str(latents_global.get_shape().as_list()))

    # latents = tf.concat([latents, latents_global], axis=-1)

    print('Final Encoder Dims: ' + str(latents.get_shape().as_list()))

    cpc_e_loss, cpc_logits, cpc_context, latent_loss = CPC(latents, target_dim=cpc_target_dim, emb_scale=1.0, scope='cpc')

    reconstructed_patch = decoder(cpc_context, anchor_layer=anchor_layer, activation='relu', norm='layer',
                                  b_train=b_train, scope='decoder')

    reconstructed_patch = tf.squeeze(reconstructed_patch, axis=[0])

    print('Reconstructed Patch Dims: ' + str(reconstructed_patch.get_shape().as_list()))

    r_loss = get_residual_loss(reconstructed_patch, Y, type='l2')
    grad_loss = get_gradient_loss(reconstructed_patch, Y)
    diff_loss = get_diff_loss(reconstructed_patch, Y, Z)
    alpha = 0.1
    beta = 0.7
    total_loss = beta * r_loss * (alpha + diff_loss) + (1 - beta) * grad_loss

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Model load failed.')
            return

        sequence_num = 0
        trX = os.listdir(test_data)
        total_num_batches = len(trX) - batch_size
        print('Total Test Number of Batches: ' + str(total_num_batches))

        valid_status = np.zeros(num_context_patches_width * num_context_patches_height)
        prev_patch_batch = []

        for start in range(total_num_batches):
            if len(prev_patch_batch) > 0:
                prev_patch_batch.pop(0)

            set_size = batch_size - len(prev_patch_batch)
            if start > 0:
                img_batches = get_image_batches(test_data, start + batch_size, set_size)
            else:
                img_batches = get_image_batches(test_data, start, set_size)

            # Create patches. batch_size * num_context_patches * num_context_patches * channel
            for i in range(set_size):
                patches = prepare_patches(img_batches[i], patch_size=[patch_height, patch_width],
                                          patch_dim=[num_context_patches_height, num_context_patches_width],
                                          stride=patch_height // 2, add_noise=False)
                prev_patch_batch.append(patches)

            patch_batch = np.array(prev_patch_batch)
            patch_batch = np.stack(patch_batch, axis=1)
            patch_batch = np.expand_dims(patch_batch, axis=-1)

            entropy_loss_list = []
            residual_loss_list = []
            gradient_loss_list = []
            diff_loss_list = []
            index_list = []
            score_list = []
            latent_loss_list = []

            for i in in_list:
                # Check validity
                b_valid = check_patch_changeness(patch_batch[i], i, valid_status[i], threshold=patch_changeness_threshold, train_mode=False)
                valid_status[i] = b_valid

                if b_valid == 0:
                    continue

                l1, l2, l3, l4 = sess.run([cpc_e_loss, r_loss, diff_loss, latent_loss],
                                      feed_dict={X: patch_batch[i],
                                                 Y: patch_batch[i][-1],
                                                 Z: patch_batch[i][-2 - predict_skip],
                                                 b_train: False})

                r_patch = sess.run([reconstructed_patch], feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: False})
                cv2.imwrite(str(start+batch_size) + '_patch_' + str(i) + '.jpg', patch_batch[i][-1])
                cv2.imwrite(str(start+batch_size) + '_r_patch_' + str(i) + '.jpg', r_patch[0])

                score = l2
                score_list.append(score)
                entropy_loss_list.append(l1)
                residual_loss_list.append(l2)
                diff_loss_list.append(l3)
                latent_loss_list.append(l4)
                index_list.append(i)

            for s in range(len(score_list)):
                #if score_list[s] > anomaly_score:
                    print('Sequence ' + str(sequence_num + batch_size) + ', Patch ' + str(index_list[s]) +
                          ', score: ' + str(score_list[s]) +
                          ', entropy loss: ' + str(entropy_loss_list[s]) +
                          ', residual loss: ' + str(residual_loss_list[s]) +
                          ', diff loss: ' + str(diff_loss_list[s]) +
                          ', latent loss: ' + str(latent_loss_list[s]))

            sequence_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/finetune/test', default='train')
    parser.add_argument('--model_path', type=str, help='Model check point file path', default='./model/m.ckpt')
    parser.add_argument('--base_model_path', type=str, help='Base Model check point file path', default='./model/m.ckpt')
    parser.add_argument('--task_model_path', type=str, help='Task Model check point file path', default='./task_model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='Training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='Test data directory', default='./test_data')
    parser.add_argument('--cold_data', type=str, help='Cold data directory', default='./cold_data')

    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path

    train_data = args.train_data
    test_data = args.test_data
    cold_data = args.cold_data

    # Input Data Dimension
    input_height = 256
    input_width = 480
    num_channel = 1
    patch_width = 32
    patch_height = 32
    num_context_patches_width = 29
    num_context_patches_height = 15

    # Dense Conv Block Base Channel Depth
    dense_block_depth = 64

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 32

    # Number of predictions: batch_size - ar_lstm_sequence_length
    batch_size = 8  # It should be larger than sequence length

    # CPC Encoding latent dimension
    representation_dim = 1024
    ar_lstm_sequence_length = 4
    ar_context_dim = 1024
    ar_lstm_hidden_layer_dims = ar_context_dim
    cpc_target_dim = 64

    # Training parameter
    num_epoch = 100

    predict_skip = 2

    pixel_brightness_threshold = 200.0
    patch_changeness_threshold = 25.0
    #patch_changeness_threshold = 40.0
    anomaly_score = 100.0

    # (x=960, y=160)
    global_context_area = [440, 640]
    global_area_height = 440 // 4
    global_area_width = 640 // 4
    # [h[0]:h[1], w[0]:w[1]]
    crop_global_area = [160, 600, 960, 1600]

    # Input Case 2
    in_list = list(range(38, 41))
    in_list += list(range(45, 48))
    in_list += [49, 79, 93, 121, 138, 149, 168, 178, 197, 206, 227, 234, 257, 263, 286, 291, 316, 320, 348]
    in_list += list(range(65, 66))
    in_list += list(range(68, 70))
    in_list += list(range(74, 76))
    in_list += list(range(96, 98))
    in_list += list(range(104, 106))
    in_list += list(range(125, 127))
    in_list += list(range(133, 135))
    in_list += list(range(162, 164))
    in_list += list(range(182, 184))
    in_list += list(range(191, 194))
    in_list += list(range(210, 212))
    in_list += list(range(221, 223))
    in_list += list(range(239, 241))
    in_list += list(range(250, 252))
    in_list += list(range(267, 270))
    in_list += list(range(279, 282))
    in_list += list(range(296, 298))
    in_list += list(range(308, 311))
    in_list += list(range(353, 355))
    in_list += list(range(382, 384))
    in_list += list(range(337, 340))
    in_list += list(range(366, 370))
    in_list += list(range(395, 398))
    in_list += list(range(374, 376))

    # Input Case 3
    # in_list = list(range(101, 110))
    # in_list = in_list + list(range(129, 140))
    # in_list = in_list + list(range(158, 170))
    # in_list = in_list + list(range(190, 198))
    # in_list = in_list + list(range(219, 226))
    # in_list = in_list + list(range(249, 254))

    # Average pixel intensity check with cold images.
    cold_img_batches = get_image_batches(cold_data, 125, 20)
    cold_mean_image = np.mean(cold_img_batches, axis=0)

    cold_patches = prepare_patches(cold_mean_image, patch_size=[patch_height, patch_width],
                                   patch_dim=[num_context_patches_height, num_context_patches_width],
                                   stride=patch_height // 2, add_noise=False)
    cold_patches = np.expand_dims(cold_patches, axis=-1)
    print('cold patches: ' + str(cold_patches.shape))

    patch_pixel_mean = cold_patches

    if mode == 'train':
        # Train unsupervised CPC encoder.
        train(model_path)
    elif mode == 'finetune':
        base_model_path = args.base_model_path
        task_model_path = args.task_model_path
        fine_tune(base_model_path, task_model_path)
    elif mode == 'test':
        num_epoch = 1
        test(model_path)
