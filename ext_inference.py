import numpy as np
import tensorflow as tf
from caps_network import Caps3d
import ext_config as config
from scipy.misc import imread
from cv2 import resize
import os
from skvideo.io import vread, vwrite
from scipy.misc import imresize
from datasets.minetto_gen import Minetto_Gen

def data_gen():
    minetto_gen = Minetto_Gen()
    while minetto_gen.has_data():
        name, video, _ = minetto_gen.get_next_video()
        yield name, video

def inference():    
    output_dir = './inference/outputs/'
    
    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.network_save_dir)
        
        for name, video in data_gen():        
            # print('Saving Cropped Video')
            # vwrite(output_dir+name+'_cropped.avi', video)

            n_frames = video.shape[0]
            segmentation_output = np.zeros((n_frames, config.vid_h, config.vid_w, 1))
            f_skip = config.frame_skip
            pred = []

            for i in range(0, n_frames, 8*f_skip):
                # if frames are skipped (subsampled) during training, they should also be skipped at test time
                # creates a batch of video clips
                x_batch = [[] for i in range(f_skip)]
                for k in range(f_skip*8):
                    if i + k >= n_frames:
                        x_batch[k % f_skip].append(np.zeros_like(video[-1]))
                    else:
                        x_batch[k % f_skip].append(video[i+k])
                x_batch = [np.stack(x, axis=0) for x in x_batch]

                # runs the network to get segmentations
                batch_pred, seg_out = sess.run([capsnet.digit_preds, capsnet.segment_layer_sig], 
                                         feed_dict={capsnet.x_input: x_batch,
                                         capsnet.is_train: False,
                                         capsnet.y_input: np.ones((f_skip,), np.int32)*-1})
                
                norm_mean = np.mean(batch_pred, axis=0)
                batch_pred_arg = np.argmax(norm_mean)
                print('(Batch) Predictions for', name, 'is', config.labels[batch_pred_arg])
                pred.append(norm_mean)

                # collects the segmented frames into the correct order
                for k in range(f_skip * 8):
                    if i + k >= n_frames:
                        continue

                    segmentation_output[i+k] = seg_out[k % f_skip][k//f_skip]

            pred_mean = np.mean(np.stack(pred, axis=0), axis=0)
            video_label = config.labels[np.argmax(pred_mean)]
            print('Video Prediction for', name + ':', video_label)

            # Final segmentation output
            segmentation_output = (segmentation_output >= 0.5).astype(np.int32)

            # Highlights the video based on the segmentation
            alpha = 0.5
            color = np.zeros((3,)) + [0.0, 0, 1.0]
            masked_vid = np.where(np.tile(segmentation_output, [1, 1, 3]) == 1,
                                  video * (1 - alpha) + alpha * color, video)

            print('Saving Segmented Video\n')
            vwrite(output_dir+name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))


if __name__=='__main__':
    inference()
