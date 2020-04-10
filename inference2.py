import numpy as np
import tensorflow as tf
from caps_network import Caps3d
import config2 as config
from scipy.misc import imread
from cv2 import resize
import os
from skvideo.io import vread, vwrite
from scipy.misc import imresize

def data_gen():
    file_loc = './inference/inputs/'
    for vid in os.listdir(file_loc):
        video_orig = vread(file_loc + vid)
        n_frames = video_orig.shape[0]
        video = np.zeros((n_frames, config.vid_h, config.vid_w, 3), dtype=np.uint8)
        for idx in range(n_frames):
            video[idx] = resize(video_orig[idx], (config.vid_w, config.vid_h))
        yield os.path.splitext(vid)[0], video/255.

    frames_dir = '../SynthVideo/out/Frames/'
    video_dir_list = ['1/18020_tr_r_l_/',
                      '3/8541_tr_b_t_/',
                      '5/19958_roll_c_/',
                      '7/11102_Panrev1_/',
                      '7/15155_Panrev1_/',
                      '8/16916_tilt1_/',
                      '10/13774_zoomout_/',
                      '10/16716_zoomout_/',
                      '10/13961_zoomout_/',
                      '10/19740_zoomout_/',
                      '11/19722_zoomin_/',]
    '''
    # Training files
    video_dir_list = ['2/19310_tr_t_b_/',
                      '7/12213_Panrev1_/',
                      '2/14451_tr_t_b_/',
                      '0/15847_tr_l_r_/',
                      '10/16454_zoomout_/',]
    '''
    for video_dir in video_dir_list:
        n_frames = len(os.listdir(frames_dir+video_dir))
        frame_start = 0
        im0 = imread(frames_dir + video_dir + ('frame_%d.jpg' % frame_start))
        _, _, ch = im0.shape
        video = np.zeros((n_frames, config.vid_h, config.vid_w, ch), dtype=np.uint8)
        for idx in range(n_frames):
            frame = imread(frames_dir + video_dir + ('frame_%d.jpg' % idx))
            frame = resize(frame, (config.vid_w, config.vid_h))
            video[idx] = frame
        yield os.path.basename(video_dir[:-1]), video/255.
        

def inference():    
    output_dir = './inference/outputs/'
    
    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.save_file_name)
        
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
            print('Video Prediction for', name + ':', video_label, '\n')

            # Final segmentation output
            segmentation_output = (segmentation_output >= 0.5).astype(np.int32)

            # Highlights the video based on the segmentation
            alpha = 0.5
            color = np.zeros((3,)) + [0.0, 0, 1.0]
            masked_vid = np.where(np.tile(segmentation_output, [1, 1, 3]) == 1,
                                  video * (1 - alpha) + alpha * color, video)

            print('Saving Segmented Video')
            vwrite(output_dir+name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))


if __name__=='__main__':
    inference()
