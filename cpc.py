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


def load_images_from_folder(folder, use_augmentation=False, add_noise=False):
    images = []

    # To Do
    # Color, Brightness Augmentation

    for filename in os.listdir(folder):
        fullname = os.path.join(folder, filename).replace("\\", "/")
        jpg_img = cv2.imread(fullname)
        img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)  # To RGB format
        img = cv2.resize(img, dsize=(input_height, input_width))

        if img is not None:
            img = np.array(img)

            n_img = img / 255.0
            images.append(n_img)

    return np.array(images)


def auto_regressive(latents, sequence_length=4, out_dim=256, scope='ar'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # [Batch Size, Latent Dims]
        sequence_batch = tf.stack([tf.slice(latents, [i, 0], [sequence_length, -1]) for i in range(batch_size - sequence_length)], axis=0)
        print('Sequence Batch Shape: ' + str(sequence_batch.get_shape().as_list()))

        context = layers.bi_lstm_network(sequence_batch, lstm_hidden_size_layer=ar_lstm_hidden_layer_dims, lstm_latent_dim=out_dim)

    return context


def CPC(latents, target_dim=64, emb_scale=0.1, scope='cpc'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        context = auto_regressive(latents, sequence_length=ar_lstm_sequence_length, out_dim=ar_context_dim)
        # [num_predict, ar_context_dim]
        print('AR Context Shape: ' + str(context.get_shape().as_list()))

        context = tf.reshape(context, shape=[1, 1, context.get_shape().as_list()[0], context.get_shape().as_list()[1]])
        context = layers.conv(context, scope='context_conv', filter_dims=[1, 1, target_dim], stride_dims=[1, 1],
                              non_linear_fn=None, bias=True)
        context = tf.squeeze(context, [0, 1])
        print('Context Shape: ' + str(context.get_shape().as_list()))
        context = context * emb_scale

        latents = tf.reshape(latents, shape=[1, 1, latents.get_shape().as_list()[0], latents.get_shape().as_list()[1]])
        targets = layers.conv(latents, scope='latent_conv', filter_dims=[1, 1, target_dim], stride_dims=[1, 1],
                              non_linear_fn=None, bias=True)
        targets = tf.squeeze(targets, [0, 1])
        print('Target Shape: ' + str(targets.get_shape().as_list()))

        logits = tf.matmul(context, targets, transpose_b=True)
        print('Logit Shape: ' + str(logits.get_shape().as_list()))

        # One Hot Label
        onehot_labels = []

        for i in range(batch_size - ar_lstm_sequence_length):
            target_index = i + ar_lstm_sequence_length
            onehot = np.zeros(batch_size)
            onehot[target_index] = 1
            onehot_labels.append(onehot)

        onehot_labels = np.array(onehot_labels)
        onehot_labels = tf.constant(onehot_labels)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

        return loss, logits


def add_residual_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, norm='layer',
                       b_train=False, use_residual=True, scope='residual_block', use_dilation=False,
                       sn=False, use_bottleneck=False):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        bn_depth = num_channel_in

        if use_bottleneck is True:
            bn_depth = num_channel_in // (num_layers * 2)
            #bn_depth = bottleneck_depth

            l = layers.conv(l, scope='bt_conv1', filter_dims=[1, 1, bn_depth], stride_dims=[1, 1],
                            dilation=[1, 1, 1, 1],
                            non_linear_fn=None, bias=False, sn=False)

        for i in range(num_layers):
            l = layers.add_residual_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=act_func, norm=norm, b_train=b_train,
                                          scope='layer' + str(i), dilation=dilation, sn=sn)

        if use_bottleneck is True:
            l = layers.conv(l, scope='bt_conv2', filter_dims=[1, 1, num_channel_in], stride_dims=[1, 1],
                            dilation=[1, 1, 1, 1],
                            non_linear_fn=None, bias=False, sn=False)

        if use_residual is True:
            l = tf.add(l, in_layer)

    return l


def add_residual_dense_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, norm='layer', b_train=False,
                             scope='residual_dense_block', use_dilation=False, stochastic_depth=False,
                             stochastic_survive=0.9):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        bn_depth = num_channel_in // (num_layers * 2)
        #bn_depth = bottleneck_depth

        l = layers.conv(l, scope='bt_conv', filter_dims=[1, 1, bn_depth], stride_dims=[1, 1], dilation=[1, 1, 1, 1],
                    non_linear_fn=None, bias=False, sn=False)

        for i in range(num_layers):
            l = layers.add_dense_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=act_func, norm=norm, b_train=b_train,
                                       scope='layer' + str(i), dilation=dilation)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, num_channel_in], act_func=act_func,
                                              scope='dense_transition_1', norm=norm, b_train=b_train, use_pool=False)

        pl = tf.constant(stochastic_survive)

        def train_mode():
            survive = tf.less(pl, tf.random_uniform(shape=[], minval=0.0, maxval=1.0))
            return tf.cond(survive, lambda: tf.add(l, in_layer), lambda: in_layer)

        def test_mode():
            return tf.add(tf.multiply(pl, l), in_layer)

        if stochastic_depth == True:
            return tf.cond(b_train, train_mode, test_mode)

    return tf.add(l, in_layer)


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

        print('Input Dims: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        l = x
        l = layers.conv(l, scope='conv1', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = layers.self_attention(l, block_depth)

        for i in range(3):
            l = add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                         act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1_' + str(i))

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr1')
        print('Map Dims: ' + str(l.get_shape().as_list()))

        for i in range(4):
            l = add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_1_' + str(i), use_bottleneck=True)

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr2')
        print('Map Dims: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_2_' + str(i), use_bottleneck=True)

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr3')
        print('Map Dims: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_3_' + str(i), use_bottleneck=True)

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr4')
        print('Map Dims: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_4_' + str(i), use_bottleneck=True)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim], stride_dims=[1, 1],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr5')

        last_layer = act_func(l)

        context = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=True, scope='gp')
        print('GP Dims: ' + str(context.get_shape().as_list()))

    return context


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])

        trX = load_images_from_folder(imgs_dirname, use_augmentation=True)
        trX = trX.reshape((-1, input_height, input_width, num_channel))

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latents = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    # [Batch Size, Latent Dims]
    print('Encoder Dims: ' + str(latents.get_shape().as_list()))

    cpc_loss, cpc_logits = CPC(latents, emb_scale=1.0, scope='cpc')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(cpc_loss)
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

        for e in range(num_epoch):
            training_batches = zip(range(0, len(trX), batch_size),
                                   range(batch_size, len(trX) + 1, batch_size))
            iteration = 0

            for start, end in training_batches:
                _, l, s_logit, c_logits = sess.run([optimizer, cpc_loss, softmax_cpc_logits, cpc_logits],
                                                   feed_dict={X: trX[start:end], b_train: True})

                iteration = iteration + 1

                #if iteration % 10 == 0:
                    #print('epoch: ' + str(e) + ', loss: ' + str(l) + ', softmax: ' + str(s_logit[0]) + ', logit: ' + str(c_logits[0]))
                print('epoch: ' + str(e) + ', loss: ' + str(l))

            try:
                saver.save(sess, model_path)
            except:
                print('Save failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='Model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='Training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='Test data directory', default='./test_data')

    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path

    imgs_dirname = args.train_data
    label_directory = args.label
    test_data = args.test_data

    # Input Data Dimension
    input_height = 128 #256
    input_width = 128 #256
    num_channel = 3

    # Dense Conv Block Base Channel Depth
    dense_block_depth = 64

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 32

    # CPC Encoding latent dimension
    representation_dim = 1024
    ar_lstm_sequence_length = 4
    ar_context_dim = 1024
    ar_lstm_hidden_layer_dims = ar_context_dim // 2

    # Training parameter
    num_epoch = 10

    # Number of predictions: batch_size - ar_lstm_sequence_length
    batch_size = 12  # It should be larger than sequence length

    if mode == 'train':
        # Train unsupervised CPC encoder.
        num_epoch = 20
        train(model_path)
