# Video Frame Sequence Prediction using CPC
# Author: Seongho Baek @seongho.baek@sk.com


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

    for idx in range(len(os.listdir(folder))):
        filename = str(idx+1) + '.jpg'
        fullname = os.path.join(folder, filename).replace("\\", "/")
        jpg_img = cv2.imread(fullname)
        #img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)  # To RGB format
        img = cv2.resize(jpg_img, dsize=(input_height, input_width))

        if img is not None:
            img = np.array(img)
            n_img = img / 255.0
            images.append(img)

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


def get_gradient_loss(img1, img2):
    h, w, c = img1.get_shape().as_list()

    # h dim
    a = tf.slice(img1, [1, 0, 0], [-1, -1, -1])
    b = tf.slice(img1, [0, 0, 0], [h-1, -1, -1])
    grad_h1 = tf.subtract(a, b)

    # w dim
    a = tf.slice(img1, [0, 1, 0], [-1, -1, -1])
    b = tf.slice(img1, [0, 0, 0], [-1, w-1, -1])
    grad_w1 = tf.subtract(a, b)

    h, w, c = img2.get_shape().as_list()

    # h dim
    a = tf.slice(img2, [1, 0, 0], [-1, -1, -1])
    b = tf.slice(img2, [0, 0, 0], [h-1, -1, -1])
    grad_h2 = tf.subtract(a, b)

    # w dim
    a = tf.slice(img2, [0, 1, 0], [-1, -1, -1])
    b = tf.slice(img2, [0, 0, 0], [-1, w-1, -1])
    grad_w2 = tf.subtract(a, b)

    return tf.reduce_mean(tf.abs(tf.subtract(grad_h1, grad_h2))) + tf.reduce_mean(tf.abs(tf.subtract(grad_w1, grad_w2)))


def auto_regressive(latents, sequence_length=4, skip=0, out_dim=256, scope='ar'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # [Batch Size, Latent Dims]
        #sequence_batch = tf.stack([tf.slice(latents, [i, 0], [sequence_length, -1]) for i in range(batch_size - sequence_length)], axis=0)
        sequence_batch = tf.slice(latents, [0, 0], [sequence_length-skip, -1])
        sequence_batch = tf.expand_dims(sequence_batch, 0)
        print('Sequence Batch Shape: ' + str(sequence_batch.get_shape().as_list()) + ', skip: ' + str(skip))

        context = layers.bi_lstm_network(sequence_batch, lstm_hidden_size_layer=ar_lstm_hidden_layer_dims, lstm_latent_dim=out_dim)
        #context = layers.lstm_network(sequence_batch, lstm_hidden_size_layer=ar_lstm_hidden_layer_dims, lstm_latent_dim=out_dim)

    return context


def CPC(latents, target_dim=64, emb_scale=0.1, scope='cpc'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        context = auto_regressive(latents, sequence_length=ar_lstm_sequence_length, skip=predict_skip, out_dim=ar_context_dim)
        # [num_predict, ar_context_dim]
        print('AR Context Shape: ' + str(context.get_shape().as_list()))

        targets = tf.slice(latents, [ar_lstm_sequence_length, 0], [-1, -1])
        residual_loss = get_residual_loss(context, targets, type='l1', gamma=1.0)
        # One Hot Label
        onehot_labels = []

        for i in range(batch_size - ar_lstm_sequence_length):
            target_index = i + ar_lstm_sequence_length
            onehot = np.zeros(batch_size)
            onehot[target_index] = 1
            print('Label: ' + str(onehot))
            onehot_labels.append(onehot)

        onehot_labels = np.array(onehot_labels)
        onehot_labels = tf.constant(onehot_labels)

        num_predicts = batch_size - ar_lstm_sequence_length

        scaled_context = tf.stack([layers.fc(context, target_dim, use_bias=False, scope='pred_' + str(i)) for i in range(num_predicts)])
        scaled_context = tf.squeeze(scaled_context)
        print('Predict Shape: ' + str(scaled_context.get_shape().as_list()))

        scaled_context = scaled_context * emb_scale
        targets = latents
        targets = layers.fc(targets, target_dim, use_bias=False, scope='target')
        print('Target Shape: ' + str(targets.get_shape().as_list()))

        scaled_context = tf.expand_dims(scaled_context, 0)
        logits = tf.matmul(scaled_context, targets, transpose_b=True)
        print('Logit Shape: ' + str(logits.get_shape().as_list()))

        entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

        return entropy_loss, residual_loss, logits, context


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

        block_depth = dense_block_depth

        l = latent
        #l = layers.fc(l, 4*4*32, non_linear_fn=act_func)

        print('Decoder Input: ' + str(latent.get_shape().as_list()))
        l = tf.reshape(l, shape=[-1, 4, 4, 64])

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                   act_func=act_func, norm=norm, b_train=b_train, scope='block0_' + str(i))

        print('Decoder Block 0: ' + str(l.get_shape().as_list()))

        block_depth = dense_block_depth * 4
        # 8 x 8
        l = layers.deconv(l, b_size=1, scope='deconv1', filter_dims=[3, 3, block_depth],
                             stride_dims=[2, 2], padding='SAME', non_linear_fn=None)

        if anchor_layer is not None:
            print('Decoder Anchor: ' + str(anchor_layer.get_shape().as_list()))
            l = tf.concat([l, anchor_layer], axis=3)
            block_depth = block_depth * 2

        print('Deconvolution 1: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                   act_func=act_func, norm=norm, b_train=b_train, scope='block1_' + str(i))

        # 16 x 16
        l = layers.deconv(l, b_size=1, scope='deconv2', filter_dims=[3, 3, block_depth],
                             stride_dims=[2, 2], padding='SAME', non_linear_fn=None)

        print('Deconvolution 2: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                   act_func=act_func, norm=norm, b_train=b_train,
                                   scope='block2_' + str(i), use_dilation=True)

        block_depth = dense_block_depth
        # 32 x 32
        l = layers.deconv(l, b_size=1, scope='deconv3', filter_dims=[3, 3, block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=None)

        #if anchor_layer is not None:
        #    l = tf.concat([l, anchor_layer], axis=3)
        #    block_depth = block_depth * 2

        print('Deconvolution 3: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=4,
                                   act_func=act_func, norm=norm, b_train=b_train,
                                   scope='block3_' + str(i), use_dilation=True)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, 3], act_func=act_func, use_pool=False, norm=norm, b_train=b_train, scope='tr1')
        l = act_func(l)

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
        block_depth = dense_block_depth

        l = x
        l = layers.conv(l, scope='conv1', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False, dilation=[1, 2, 2, 1])

        l = layers.self_attention(l, block_depth)

        for i in range(3):
            l = add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                         act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1_' + str(i))

        #anchor_layer = tf.slice(l, [l.get_shape().as_list()[0]-4, 0, 0, 0], [1, -1, -1, -1])

        # l = layers.self_attention(l, block_depth)

        block_depth = block_depth * 2

        # [16 x 16]
        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr1')
        print('Encoder Block 0: ' + str(l.get_shape().as_list()))

        for i in range(4):
            l = add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=4, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_1_' + str(i), use_bottleneck=True)

        block_depth = block_depth * 2

        # [8 x 8]
        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr2')
        #print('Map Dims: ' + str(l.get_shape().as_list()))

        #for i in range(5):
        #    l = add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
        #                           norm=norm, b_train=b_train, scope='res_block_2_' + str(i), use_bottleneck=True)

        #block_depth = block_depth * 2

        #l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
        #                                      act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
        #                                      scope='tr3')
        print('Encoder Block 1: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_3_' + str(i), use_bottleneck=True)

        anchor_layer = tf.slice(l, [l.get_shape().as_list()[0] - predict_skip - 1, 0, 0, 0], [1, -1, -1, -1])

        block_depth = block_depth * 2

        l = layers.add_dense_transition_layer(l, filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr4')
        print('Encoder Block 2: ' + str(l.get_shape().as_list()))

        for i in range(5):
            l = add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                   norm=norm, b_train=b_train, scope='res_block_4_' + str(i), use_bottleneck=True)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim], stride_dims=[1, 1],
                                              act_func=act_func, norm=norm, b_train=b_train, use_pool=False,
                                              scope='tr5')

        last_layer = act_func(l)

        context = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=True, scope='gp')
        print('Encoder GP Dims: ' + str(context.get_shape().as_list()))

    return context, anchor_layer


def prepare_patches(image, patch_size=[24, 24], patch_dim=[7, 7], stride=12):
    patches = []
    patch_w = patch_size[0]
    patch_h = patch_size[1]

    #cv2.imwrite('original.jpg', image*255.0)
    #print('image shape: ', image.shape)
    for h in range(patch_dim[0]):
        for w in range(patch_dim[1]):
            #print('h:', h*stride, ' w: ', h*stride + patch_h)
            #print('w:', w*stride, ' w: ', w*stride + patch_w)
            patch = image[h*stride:(h*stride + patch_h), w*stride:(w*stride + patch_w)].copy()
            #sample = patch*255.0
            #sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            #cv2.imwrite('patch_' + str(h) + '.jpg', sample)
            #print('Patch dims: ', patch.shape)
            patches.append(patch)

    #print('Num patches: ', len(patches))
    return np.array(patches)


def fine_tune(base_model_path, task_model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])

        trX = load_images_from_folder(imgs_dirname, use_augmentation=True)
        trX = trX.reshape((-1, input_height, input_width, num_channel))

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latents, anchor_layer = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    # [Batch Size, Latent Dims]
    print('Encoder Dims: ' + str(latents.get_shape().as_list()))

    cpc_e_loss, cpc_r_loss, cpc_logits, cpc_context = CPC(latents, target_dim=cpc_target_dim, emb_scale=1.0,
                                                          scope='cpc')

    reconstructed_patch = decoder(cpc_context, anchor_layer=anchor_layer, activation='relu', norm='layer',
                                  b_train=b_train, scope='decoder')

    reconstructed_patch = tf.squeeze(reconstructed_patch)

    print('Reconstructed Patch Dims: ' + str(reconstructed_patch.get_shape().as_list()))

    r_loss = get_residual_loss(reconstructed_patch, Y, type='l1')
    grad_loss = get_gradient_loss(reconstructed_patch, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(cpc_e_loss + r_loss + grad_loss)
    softmax_cpc_logits = tf.nn.softmax(logits=cpc_logits)

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
                                        or v.name.split('/')[0] == 'cpc']
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, base_model_path)
                print('Partial Model Restored. Start fine tune. Wait ...')
            except:
                print('Start New Training. Wait ...')

        batch_start = 0
        itr_save = 0

        for e in range(num_epoch):
            training_batches = zip(range(batch_start, len(trX), batch_size),
                                   range(batch_size, len(trX) + 1, batch_size))

            #if e > batch_size:
            #    training_batches = shuffle(training_batches)
            training_batches = shuffle(training_batches)
            batch_start += 1
            total_num_batches = len(training_batches)

            for start, end in training_batches:
                patch_batch = []

                # Create patches. batch_size * num_context_patches * num_context_patches * channel
                for i in range(batch_size):
                    patches = prepare_patches(trX[start + i], patch_size=[patch_height, patch_width],
                                              patch_dim=[num_context_patches, num_context_patches],
                                              stride=patch_height // 2)
                    # [49, 32, 32, 3]
                    patch_batch.append(patches)

                # [8, 49, 32, 32, 3]
                patch_batch = np.array(patch_batch)
                # [49, 8, 32, 32, 3]
                patch_batch = np.stack(patch_batch, axis=1)

                loss_list = []
                g_loss_list = []
                index_list = []

                for i in range(num_context_patches * num_context_patches):
                    if i not in out_list:
                        _, residual_loss, gradient_loss = sess.run(
                            [optimizer, r_loss, grad_loss],
                            feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: True})
                        # r_patch = sess.run(
                        #    [reconstructed_patch],
                        #    feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: True})

                        loss_list.append(residual_loss)
                        g_loss_list.append(gradient_loss)
                        index_list.append(i)

                    # if i % ((num_context_patches * num_context_patches)//2) == 0:
                    # print('epoch: ' + str(e) + ', entropy loss: ' + str(l1) + ', reconstruct loss: ' + str(l3))
                    #    print('epoch: ' + str(e) + ', frame: ' + str(end) + ', patch ' + str(i) + ', entropy loss: ' + str(l1) + ', reconstruct loss: ' + str(l3))
                    # cv2.imwrite(str(end) + '_patch_' + str(i) + '.jpg', patch_batch[i][-1])
                    # cv2.imwrite(str(end) + '_r_patch_' + str(i) + '.jpg', r_patch[0])

                max_index = loss_list.index(max(loss_list))
                patch_index = index_list[max_index]

                itr_save += 1
                print('Epoch: ' + str(e) + ', ' + str(itr_save) + '/' + str(total_num_batches) +
                      ', Frame: ' + str(end) +
                      ', Patch #' + str(patch_index) + ': ' + str(loss_list[max_index]) +
                      ', ' + str(g_loss_list[max_index]))

                if itr_save % 30 == 0:
                    try:
                        saver = tf.train.Saver()
                        saver.save(sess, task_model_path)
                        itr_save = 0
                    except:
                        print('Save failed')


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])

        trX = load_images_from_folder(imgs_dirname, use_augmentation=True)
        trX = trX.reshape((-1, input_height, input_width, num_channel))

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latents, anchor_layer = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    # [Batch Size, Latent Dims]
    print('Encoder Dims: ' + str(latents.get_shape().as_list()))

    cpc_e_loss, cpc_r_loss, cpc_logits, cpc_context = CPC(latents, target_dim=cpc_target_dim, emb_scale=1.0, scope='cpc')

    reconstructed_patch = decoder(cpc_context, anchor_layer=anchor_layer, activation='relu', norm='layer', b_train=b_train, scope='decoder')

    reconstructed_patch = tf.squeeze(reconstructed_patch)

    print('Reconstructed Patch Dims: ' + str(reconstructed_patch.get_shape().as_list()))

    r_loss = get_residual_loss(reconstructed_patch, Y, type='l1')
    grad_loss = get_gradient_loss(reconstructed_patch, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(cpc_e_loss + r_loss + grad_loss)
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

        batch_start = 0
        itr_save = 0

        for e in range(num_epoch):
            training_batches = zip(range(batch_start, len(trX), batch_size),
                                   range(batch_size, len(trX) + 1, batch_size))

            total_num_batches = len(training_batches)

            #if e > batch_size:
            #    training_batches = shuffle(training_batches)
            training_batches = shuffle(training_batches)
            batch_start += 1

            for start, end in training_batches:
                patch_batch = []

                # Create patches. batch_size * num_context_patches * num_context_patches * channel
                for i in range(batch_size):
                    patches = prepare_patches(trX[start + i], patch_size=[patch_height, patch_width],
                                              patch_dim=[num_context_patches, num_context_patches], stride=patch_height//2)
                    # [49, 32, 32, 3]
                    patch_batch.append(patches)

                # [8, 49, 32, 32, 3]
                patch_batch = np.array(patch_batch)
                # [49, 8, 32, 32, 3]
                patch_batch = np.stack(patch_batch, axis=1)

                loss_list = []
                g_loss_list = []
                index_list = []

                for i in range(num_context_patches * num_context_patches):
                    if i not in out_list:
                        _, residual_loss, gradient_loss = sess.run(
                            [optimizer, r_loss, grad_loss],
                            feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: True})
                            #r_patch = sess.run(
                            #    [reconstructed_patch],
                            #    feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: True})

                        loss_list.append(residual_loss)
                        g_loss_list.append(gradient_loss)
                        index_list.append(i)

                    #if i % ((num_context_patches * num_context_patches)//2) == 0:
                        # print('epoch: ' + str(e) + ', entropy loss: ' + str(l1) + ', reconstruct loss: ' + str(l3))
                    #    print('epoch: ' + str(e) + ', frame: ' + str(end) + ', patch ' + str(i) + ', entropy loss: ' + str(l1) + ', reconstruct loss: ' + str(l3))
                        # cv2.imwrite(str(end) + '_patch_' + str(i) + '.jpg', patch_batch[i][-1])
                        # cv2.imwrite(str(end) + '_r_patch_' + str(i) + '.jpg', r_patch[0])

                max_index = loss_list.index(max(loss_list))
                patch_index = index_list[max_index]
                itr_save += 1

                print('Epoch: ' + str(e) + ', ' + str(itr_save) + '/' + str(total_num_batches) +
                      ', Frame: ' + str(end) +
                      ', Patch #' + str(patch_index) + ': ' + str(loss_list[max_index]) +
                      ', ' + str(g_loss_list[max_index]))

                if itr_save % 30 == 0:
                    try:
                        saver.save(sess, model_path)
                        itr_save = 0
                    except:
                        print('Save failed')


def test(model_path):
    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [patch_height, patch_width, num_channel])

        trX = load_images_from_folder(test_data, use_augmentation=True)
        trX = trX.reshape((-1, input_height, input_width, num_channel))

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latents, anchor_layer = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    # [Batch Size, Latent Dims]
    print('Encoder Dims: ' + str(latents.get_shape().as_list()))

    cpc_e_loss, cpc_r_loss, cpc_logits, cpc_context = CPC(latents, target_dim=cpc_target_dim, emb_scale=1.0, scope='cpc')

    reconstructed_patch = decoder(cpc_context, anchor_layer=anchor_layer, activation='relu', norm='layer', b_train=b_train, scope='decoder')

    reconstructed_patch = tf.squeeze(reconstructed_patch)

    print('Reconstructed Patch Dims: ' + str(reconstructed_patch.get_shape().as_list()))

    r_loss = get_residual_loss(reconstructed_patch, Y, type='l1')
    grad_loss = get_gradient_loss(reconstructed_patch, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(cpc_e_loss + r_loss)
    softmax_cpc_logits = tf.nn.softmax(logits=cpc_logits)

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

        test_batches = zip(range(0, len(trX), batch_size),
                           range(batch_size, len(trX) + 1, batch_size))

        sequence_num = 0

        for start, end in test_batches:
            sequence_num += 1
            patch_batch = []

            # Create patches. batch_size * num_context_patches * num_context_patches * channel
            for i in range(batch_size):
                patches = prepare_patches(trX[start + i], patch_size=[patch_height, patch_width],
                                          patch_dim=[num_context_patches, num_context_patches],
                                          stride=patch_height // 2)
                patch_batch.append(patches)

            patch_batch = np.array(patch_batch)
            patch_batch = np.stack(patch_batch, axis=1)

            #print(patch_batch.shape)

            residual_loss_list = []
            entropy_loss_list = []
            gradient_loss_list = []
            index_list = []
            score_list = []

            for i in range(num_context_patches * num_context_patches):
                if i not in out_list:
                    l1, l2, l3 = sess.run([cpc_e_loss, r_loss, grad_loss],
                                          feed_dict={X: patch_batch[i], Y: patch_batch[i][-1], b_train: False})
                    score = l1 + l2
                    score_list.append(score)
                    entropy_loss_list.append(l1)
                    residual_loss_list.append(l2)
                    gradient_loss_list.append(l3)
                    index_list.append(i)

            score_min = min(score_list)
            score_max = max(score_list)

            norm_score_list = [score_max * (s - score_min)/(score_max - score_min + 0.1) for s in score_list]

            for s in range(len(norm_score_list)):
                print('Sequence ' + str(sequence_num) + ', Patch ' + str(index_list[s]) +
                      ', score: ' + str(norm_score_list[s]) +
                      ', entropy loss: ' + str(entropy_loss_list[s]) +
                      ', residual loss: ' + str(residual_loss_list[s]) +
                      ', gradient loss: ' + str(gradient_loss_list[s]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/finetune/test', default='train')
    parser.add_argument('--model_path', type=str, help='Model check point file path', default='./model/m.ckpt')
    parser.add_argument('--base_model_path', type=str, help='Base Model check point file path', default='./model/m.ckpt')
    parser.add_argument('--task_model_path', type=str, help='Task Model check point file path', default='./task_model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='Training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='Test data directory', default='./test_data')

    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path

    imgs_dirname = args.train_data
    test_data = args.test_data

    # Input Data Dimension
    input_height = 256
    input_width = 256
    num_channel = 3
    patch_width = 32
    patch_height = 32
    num_context_patches = 15

    # Dense Conv Block Base Channel Depth
    dense_block_depth = 64

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 32

    # Number of predictions: batch_size - ar_lstm_sequence_length
    batch_size = 8  # It should be larger than sequence length

    # CPC Encoding latent dimension
    representation_dim = 1024
    ar_lstm_sequence_length = batch_size - 1
    ar_context_dim = 1024
    ar_lstm_hidden_layer_dims = ar_context_dim
    cpc_target_dim = 512

    # Training parameter
    num_epoch = 100

    predict_skip = 2

    out_list = list(range(35))
    out_list = out_list + list(range(164, 225))
    out_list = out_list + [42, 43, 44, 45, 46, 47, 60, 61, 75, 90, 105, 120, 121, 135, 136, 137, 138, 150, 151, 152, 153, 154]

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
