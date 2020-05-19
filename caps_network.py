import config
import gc
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import time
import numpy as np
from caps_layers import create_prim_conv3d_caps, create_dense_caps, layer_shape, create_conv3d_caps
from tqdm import tqdm


def create_skip_connection(in_caps_layer, n_units, kernel_size, strides=(1, 1, 1), 
                            padding='VALID', name='skip', activation=tf.nn.relu):
    '''
    skip_connection1 = create_skip_connection(sec_caps, 128, kernel_size=[1, 3, 3], 
                                                    strides=[1, 1, 1], padding='SAME', 
                                                    name='skip_1')
    '''

    in_caps_layer = in_caps_layer[0]
    batch_size = tf.shape(in_caps_layer)[0]
    _, d, h, w, ch, _ = in_caps_layer.get_shape()
    d, h, w, ch = map(int, [d, h, w, ch])

    in_caps_res = tf.reshape(in_caps_layer, [batch_size, d, h, w, ch * 16])

    return tf.layers.conv3d_transpose(in_caps_res, n_units, kernel_size=kernel_size, 
                                        strides=strides, padding=padding, use_bias=False, 
                                        activation=activation, name=name)


class Caps3d(object):
    def __init__(self,  input_shape=(None, 8, config.vid_h, config.vid_w, 3)):
        self.input_shape = input_shape
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_weights()

            # inputs to the network
            #with tf.device('/gpu:0'):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=self.input_shape)
            self.y_input = tf.placeholder(dtype=tf.int32, shape=[None])
            self.y_bbox = tf.placeholder(dtype=tf.float32, shape=(None, 8, len(config.ann_types), config.vid_h, config.vid_w, 1))
            self.is_train = tf.placeholder(tf.bool)
            self.is_real = tf.placeholder(tf.bool)
            self.m = tf.placeholder(tf.float32, shape=())
            self.real_data_flag = True if config.data_type=='real' else False

            # initializes the network
            self.save_variable_list = []
            self.init_network(self.save_variable_list)

            # initializes the loss
            self.cur_m = config.start_m
            self.init_loss_and_opt()

            # initializes the saver
            self.saver = tf.train.Saver()

    def init_weights(self):
        self.w_and_b = {
            'none': None,
            'zero': tf.zeros_initializer()
        }

    def init_network(self, save_variable_list):
        print('Building Caps3d Model')

        # creates the video encoder
        self.conv1 = tf.layers.conv3d(self.x_input, 64, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv1')
        # save_variable_list.append(conv1)
        
        self.conv2 = tf.layers.conv3d(self.conv1, 128, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv2')
        # save_variable_list.append(conv2)
        
        self.conv3 = tf.layers.conv3d(self.conv2, 64, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv3')
        # save_variable_list.append(conv3)

        self.conv4 = tf.layers.conv3d(self.conv3, 128, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv4')
        # save_variable_list.append(conv4)

        self.conv5 = tf.layers.conv3d(self.conv4, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv5')
        # save_variable_list.append(conv5)

        self.conv6 = tf.layers.conv3d(self.conv5, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv6')
        # save_variable_list.append(conv6)

        self.conv7 = tf.layers.conv3d(self.conv6, 512, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv7')
        # save_variable_list.append(conv7)

        self.conv8 = tf.layers.conv3d(self.conv7, 512, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                                 bias_initializer=self.w_and_b['zero'], name='conv8')
        # save_variable_list.append(conv8)
        
        if config.print_layers:
            print('Conv1:', self.conv1.get_shape())
            print('Conv2:', self.conv2.get_shape())
            print('Conv3:', self.conv3.get_shape())
            print('Conv4:', self.conv4.get_shape())
            print('Conv5:', self.conv5.get_shape())
            print('Conv6:', self.conv6.get_shape())
            print('Conv7:', self.conv7.get_shape())
            print('Conv8:', self.conv8.get_shape())

        # with tf.device('/gpu:0'):
        # creates the primary capsule layer: conv caps1
        self.prim_caps = create_prim_conv3d_caps(self.conv8, 32, kernel_size=[3, 9, 9], strides=[1, 1, 1],                         
                                            padding='VALID', name='prim_caps')
        # save_variable_list.extend(prim_caps)
        
        # with tf.device('/gpu:0'):
        # creates the secondary capsule layer: conv caps2
        sec_caps = create_conv3d_caps(self.prim_caps, 32, kernel_size=[3, 5, 5], strides=[1, 2, 2],
                                      padding='VALID', name='sec_caps', route_mean=True)
        # save_variable_list.extend(sec_caps)
        
        #with tf.device('/gpu:0'):
        # creates the final capsule layer: class caps
        pred_caps = create_dense_caps(sec_caps, config.n_classes, subset_routing=-1, route_min=0.0,
                                      name='pred_caps', coord_add=True, ch_same_w=True)
        # save_variable_list.extend(pred_caps)

        if config.print_layers:
            print('Primary Caps:', layer_shape(self.prim_caps))
            print('Second Caps:', layer_shape(sec_caps))
            print('Prediction Caps:', layer_shape(pred_caps))

        # obtains the activations of the class caps layer and gets the class prediction
        self.digit_preds = tf.reshape(pred_caps[1], (-1, config.n_classes))
        self.predictions = tf.cast(tf.argmax(input=self.digit_preds, axis=1), tf.int32)

        pred_caps_poses = pred_caps[0]
        batch_size = tf.shape(pred_caps_poses)[0]
        _, n_classes, dim = pred_caps_poses.get_shape()
        n_classes, dim = map(int, [n_classes, dim])

        # masks the capsules that are not the ground truth (training) or the prediction (testing)
        vec_to_use = tf.cond(tf.logical_or(self.is_train, self.is_real), lambda: self.y_input, lambda: self.predictions)
        # vec_to_use = tf.cond(self.is_train, lambda: self.y_input, lambda: self.predictions)
        vec_to_use = tf.one_hot(vec_to_use, depth=n_classes)
        vec_to_use = tf.tile(tf.reshape(vec_to_use, (batch_size, n_classes, 1)), multiples=[1, 1, dim])
        masked_caps = pred_caps_poses * tf.cast(vec_to_use, dtype=tf.float32)
        masked_caps = tf.reshape(masked_caps, (batch_size, n_classes * dim))

        # creates the decoder network
        self.recon_fc1 = tf.layers.dense(masked_caps, 4 * 10 * 24 * 1, activation=tf.nn.relu, name='recon_fc1')
        # save_variable_list.append(recon_fc1)
        recon_fc1_reshaped = tf.reshape(self.recon_fc1, (batch_size, 4, 10, 24, 1))

        self.deconv1 = tf.layers.conv3d_transpose(recon_fc1_reshaped, 128, kernel_size=[1, 3, 3], 
                                            strides=[1, 1, 1], padding='SAME', 
                                            use_bias=False, activation=tf.nn.relu, 
                                            name='deconv1')
        # save_variable_list.append(deconv1)
        
        self.skip_connection1 = create_skip_connection(sec_caps, 128, kernel_size=[1, 3, 3], 
                                                    strides=[1, 1, 1], padding='SAME', 
                                                    name='skip_1')
        # save_variable_list.append(skip_connection1)

        deconv1_concat = tf.concat([self.deconv1, self.skip_connection1], axis=-1)

        self.deconv2 = tf.layers.conv3d_transpose(deconv1_concat, 128, kernel_size=[3, 6, 6], strides=[1, 2, 2],
                                             padding='VALID', use_bias=False, activation=tf.nn.relu, name='deconv2')
        # save_variable_list.append(deconv2)
            
        self.skip_connection2 = create_skip_connection(self.prim_caps, 128, kernel_size=[1, 3, 3], strides=[1, 1, 1],
                                                  padding='SAME', name='skip_2')
        # save_variable_list.append(skip_connection2)

        print('deconv1:', self.deconv1.get_shape())                                          
        print('deconv2:', self.deconv2.get_shape())
        print('skip_connection2:', self.skip_connection2.get_shape())
        deconv2_concat = tf.concat([self.deconv2, self.skip_connection2], axis=-1)

        self.deconv3 = tf.layers.conv3d_transpose(deconv2_concat, 256, kernel_size=[3, 9, 9], strides=[1, 1, 1],
                                             padding='VALID',
                                             use_bias=False, activation=tf.nn.relu, name='deconv3')
        # save_variable_list.append(deconv3)

        self.deconv4 = tf.layers.conv3d_transpose(self.deconv3, 256, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                             use_bias=False, activation=tf.nn.relu, name='deconv4')
        # save_variable_list.append(deconv4)
        
        self.deconv5 = tf.layers.conv3d_transpose(self.deconv4, 256, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                             use_bias=False, activation=tf.nn.relu, name='deconv5')
        # save_variable_list.append(deconv5)
        
        deconv6 = tf.layers.conv3d_transpose(self.deconv5, 256, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                             use_bias=False, activation=tf.nn.relu, name='deconv6')

        self.segment_layer = {}
        self.segment_layer_sig = {}
        for ann_type in config.ann_types:
            self.segment_layer[ann_type] = tf.layers.conv3d(deconv6, 1, kernel_size=[1, 3, 3], strides=[1, 1, 1],
                                                  padding='SAME', activation=None, name='segment_layer_'+ann_type)
            self.segment_layer_sig[ann_type] = tf.nn.sigmoid(self.segment_layer[ann_type])

        if config.print_layers:
            print('Deconv Layer 1:', self.deconv1.get_shape())
            print('Deconv Layer 2:', self.deconv2.get_shape())
            print('Deconv Layer 3:', self.deconv3.get_shape())
            print('Deconv Layer 4:', self.deconv4.get_shape())
            print('Deconv Layer 5:', self.deconv5.get_shape())
            print('Deconv Layer 6:', deconv6.get_shape())
            print('Segmentation Layer:', self.segment_layer[config.ann_types[0]].get_shape())


    def init_loss_and_opt(self):
        if config.data_type == 'synth':
            y_onehot = tf.one_hot(indices=self.y_input, depth=config.n_classes)

            # get a_t
            a_i = tf.expand_dims(self.digit_preds, axis=1)
            y_onehot2 = tf.expand_dims(y_onehot, axis=2)
            a_t = tf.matmul(a_i, y_onehot2)

            # calculate spread loss
            spread_loss = tf.square(tf.maximum(0.0, self.m - (a_t - a_i)))
            spread_loss = tf.matmul(spread_loss, 1. - y_onehot2)
            self.class_loss = tf.reduce_sum(tf.reduce_sum(spread_loss, axis=[1, 2]))
        else:
            self.class_loss = tf.constant(0.0)
        
        '''
        # segmentation loss
        segment = tf.contrib.layers.flatten(self.segment_layer)
        y_bbox = tf.contrib.layers.flatten(self.y_bbox)
        self.segmentation_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_bbox, logits=segment))
        self.segmentation_loss = config.segment_coef * self.segmentation_loss
        '''
        
        # segmentation loss
        for i, ann_type in enumerate(config.ann_types):
            segment = tf.contrib.layers.flatten(self.segment_layer[ann_type])
            y_bbox = tf.contrib.layers.flatten(self.y_bbox[:, :, i, :, :, :])
            if i == 0:
                self.segmentation_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_bbox, logits=segment))
            else:     
                self.segmentation_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_bbox, logits=segment))
            
        self.segmentation_loss = config.segment_coef * self.segmentation_loss

        # accuracy of a given batch
        if config.data_type == 'synth':
            correct = tf.cast(tf.equal(self.predictions, self.y_input), tf.float32)
            self.tot_correct = tf.reduce_sum(correct)
            self.accuracy = tf.reduce_mean(correct)
        else:
            self.tot_correct = tf.shape(self.y_input)[0]
            self.accuracy = tf.constant(.99)

        if config.data_type == 'synth':
            self.total_loss = self.class_loss + self.segmentation_loss
        else:
            self.total_loss = self.segmentation_loss
        
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, name='Adam',
                                           epsilon=config.epsilon)

        self.train_op = optimizer.minimize(loss=self.total_loss)

    def train(self, sess, data_gen):
        start_time = time.time()
        # continues until no more training data is generated
        losses, batch, acc, s_losses = 0, 0, 0, 0
        # mlosses, slosses, corrs = [], [], 0
        while data_gen.has_data():
            x_batch, bbox_batch, y_batch = data_gen.get_batch(config.batch_size)
            
            # runs network on batch
            _, loss, s_loss, preds = sess.run([self.train_op, self.class_loss, 
                                               self.segmentation_loss, self.digit_preds],
                                               feed_dict={self.x_input: x_batch, 
                                                          self.y_input: y_batch,
                                                          self.m: self.cur_m, self.is_train: True,
                                                          self.is_real: True, self.y_bbox: bbox_batch})

            # accumulates loses and accuracies
            acc += np.count_nonzero(np.argmax(preds, axis=1) == np.array(y_batch))/config.batch_size
            losses += loss
            s_losses += s_loss
            batch += 1

            # prints the loss and accuracy statistics after a certain number of batches
            if batch % config.batches_until_print == 0:
                print(preds[0][:10])  # prints activations just in case of numerical instability
                print('Finished %d batches. %d(s) since start. Avg Classification Loss is %.4f. '
                      'Avg Segmentation Loss is %.4f. Accuracy is %.4f.'
                      % (batch, time.time()-start_time, losses / batch, s_losses / batch, acc / batch))
                
            del x_batch, bbox_batch, y_batch
            gc.collect()

        # prints the loss and accuracy statistics for the entire epoch
        print(preds[0][:10])  # prints activations just in case of numerical instability
        print('Epoch finished in %d(s). Avg Classification loss is %.4f. Avg Segmentation Loss is %.4f. '
              'Accuracy is %.4f.'
              % (time.time() - start_time, losses / batch, s_losses / batch,  acc / batch))

        return losses / batch, s_losses / batch, acc / batch

    def eval(self, sess, data_gen, validation=True):
        mlosses, slosses, corrs = [], [], 0
        conf_matrix = np.zeros((config.n_classes, config.n_classes), dtype=np.int32)
        start_time = time.time()
        batch = 0
        for _ in tqdm(range(data_gen.n_videos)):
            video, bbox, label = data_gen.get_next_video()

            # gets losses and predictionfor a single video
            mloss, sloss, pred = self.eval_on_vid(sess, video, bbox, label, validation)

            # accumulates video statistics
            if self.real_data_flag:
                conf_matrix[label, pred] += 1
            else:    
                conf_matrix[pred, pred] += 1
                
            mlosses.append(mloss)
            slosses.append(sloss)
            corrs += (1 if pred == label else 0)
            batch += 1

            # print statistics every 500 videos
            if batch % 500 == 0:
                print('Tested %d videos. %d(s) since start. Avg Accuracy is %.4f'
                      % (batch, time.time() - start_time, float(corrs) / batch))
                
            del video, bbox, label
            gc.collect()
            
        # print evaluation statistics for all evaluation videos
        print('Evaluation done in %d(s).' % (time.time() - start_time))
        print('Test Classification Loss: %.4f. Test Segmentation Loss: %.4f. Accuracy: %.4f.'
              % (float(np.mean(mlosses)), float(np.mean(slosses)), float(corrs) / data_gen.n_videos))

        return np.mean(mlosses), np.mean(slosses), float(corrs) / data_gen.n_videos, conf_matrix

    def eval_on_vid(self, sess, video, bbox, label, validation):
        losses, slosses, norms = [], [], []

        # ensures the video is trimmed and separate video into clips of 8 frames
        f_skip = config.frame_skip
        clips = []
        n_frames = video.shape[0]
        for i in range(0, n_frames, 8 * f_skip):
            for j in range(f_skip):
                b_vid, b_bbox = [], []
                for k in range(8):
                    ind = i + j + k * f_skip
                    if ind >= n_frames:
                        b_vid.append(np.zeros((1, config.vid_h, config.vid_w, 3), dtype=np.float32))
                        b_bbox.append(np.zeros((1, len(config.ann_types), config.vid_h, config.vid_w, 1), dtype=np.float32))
                    else:
                        b_vid.append(video[ind:ind + 1, :, :, :])
                        b_bbox.append(bbox[ind:ind + 1, :, :, :, :])

                clips.append((np.concatenate(b_vid, axis=0), np.concatenate(b_bbox, axis=0), label))
                if clips[-1][1].sum() == 0:
                    clips.pop(-1)

        if len(clips) == 0:
            print('Video has no bounding boxes')
            return 0, 0, 0

        # groups clips into batches
        batches, gt_segmentations = [], []
        for i in range(0, len(clips), config.batch_size):
            x_batch, bb_batch, y_batch = [], [], []
            for j in range(i, min(i + config.batch_size, len(clips))):
                x, bb, y = clips[j]
                x_batch.append(x)
                bb_batch.append(bb)
                y_batch.append(y)
            batches.append((x_batch, bb_batch, y_batch))
            gt_segmentations.append(np.stack(bb_batch))

        # if doing validation, only do one batch per video
        if validation:
            batches = batches[:1]

        # runs the network on the clips
        n_clips = 0
        
        for x_batch, bbox_batch, y_batch in batches:
            loss, sloss, norm = sess.run([self.class_loss, self.segmentation_loss, self.digit_preds],
                                         feed_dict={self.x_input: x_batch, self.y_input: y_batch,
                                                    self.m: 0.9, self.is_train: False,
                                                    self.is_real: self.real_data_flag, self.y_bbox: bbox_batch})

            n_clips_in_batch = len(x_batch)
            losses.append(loss * n_clips_in_batch)
            slosses.append(sloss * n_clips_in_batch)
            norms.append(norm)
            n_clips += n_clips_in_batch

        # calculates network prediction
        if len(norms) > 1:
            concat_norms = np.concatenate(norms, axis=0)
        else:
            concat_norms = norms[0]
        norm_mean = np.mean(concat_norms, axis=0)
        pred = np.argmax(norm_mean)

        # gets average losses
        fin_mloss = float(np.sum(losses) / n_clips)
        fin_sloss = float(np.sum(slosses) / n_clips)

        return fin_mloss, fin_sloss, pred


    def save(self, sess, file_name, ep):
        # saves the model
        # save_path = self.saver.save(sess, file_name, global_step=ep, write_meta_graph=True)
        saver = tf.train.Saver(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6,
                               self.conv7, self.conv8, self.prim_caps, self.recon_fc1, self.deconv1, self.skip_connection1,
                               self.deconv2, self.skip_connection2, self.deconv3, self.deconv4, self.deconv5)
        save_path = saver.save(sess, file_name, global_step=ep, write_meta_graph=True)  
        # save_path = self.saver.save(sess, config.network_save_dir + 'pretrained_capsnet_83.ckpt', global_step=ep, write_meta_graph=True)
        print("Model saved in file: %s" % save_path)


    def load(self, sess, file_dir):
        # loads the model
        file_name = tf.train.latest_checkpoint(file_dir)
        print('Restoring model from file: %s' % file_name)
        self.saver.restore(sess, file_name)
        # self.saver.restore(sess, file_name)

