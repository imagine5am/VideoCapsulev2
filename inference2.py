import numpy as np
import tensorflow as tf
from caps_network import Caps3d
import config2 as config
from scipy.misc import imread
from cv2 import resize
import os
from skvideo.io import vread, vwrite
from scipy.misc import imresize

output_dir = './inference/outputs/'
labels = {0: '_tr_l_r_',
          1: '_tr_r_l_',
          2: '_tr_t_b_',
          3: '_tr_b_t_',
          4: '_roll_ac_',
          5: '_roll_c_',
          6: '_Pan1_',
          7: '_Panrev1_',
          8: '_tilt1_',
          9: '_tilt1rev_',
          10: '_zoomout_',
          11: '_zoomin_',
          }

def inference(video, dir=False):
    name = os.path.basename(video[:-1])
    if not dir:
        name = os.path.splitext(name)[0]
        
    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.save_file_name)
        if not dir:
            video_orig = vread(video)
            n_frames = video_orig.shape[0]
            video = np.zeros((n_frames, config.vid_h, config.vid_w, 3), dtype=np.uint8)
            for idx in range(n_frames):
                frame = resize(video_orig[idx], (config.vid_w, config.vid_h))
                video[idx] = frame
        else:
            video_dir = video
            n_frames = len(os.listdir(video_dir))
            frame_start = 0
            im0 = imread(video_dir + ('frame_%d.jpg' % frame_start))
            h, w, ch = im0.shape
            video = np.zeros((n_frames, config.vid_h, config.vid_w, ch), dtype=np.uint8)
            for idx in range(n_frames):
                frame = imread(video_dir + ('frame_%d.jpg' % idx))
                frame = resize(frame, (config.vid_w, config.vid_h))
                video[idx] = frame
                
        n_frames = video.shape[0]
        crop_size = (config.vid_h, config.vid_w)
        '''
        # assumes a given aspect ratio of (240, 320). If given a cropped video, then no resizing occurs
        if video.shape[1] != 112 and video.shape[2] != 112:
            h, w = 120, 160

            video_res = np.zeros((n_frames, 120, 160, 3))

            for f in range(n_frames):
                video_res[f] = imresize(video[f], (120, 160))
        else:
            h, w = 112, 112
            video_res = video

        # crops video to 112x112
        margin_h = h - crop_size[0]
        h_crop_start = int(margin_h / 2)
        margin_w = w - crop_size[1]
        w_crop_start = int(margin_w / 2)
        video_cropped = video_res[:, h_crop_start:h_crop_start+crop_size[0], w_crop_start:w_crop_start+crop_size[1], :]
        print('Saving Cropped Video')
        vwrite('cropped.avi', video_cropped)
        '''
        print('Saving Cropped Video')
        vwrite(output_dir+name+'_cropped.avi', video)

        video_cropped = video/255.

        segmentation_output = np.zeros((n_frames, crop_size[0], crop_size[1], 1))
        f_skip = config.frame_skip
        pred = []

        for i in range(0, n_frames, 8*f_skip):
            # if frames are skipped (subsampled) during training, they should also be skipped at test time
            # creates a batch of video clips
            x_batch = [[] for i in range(f_skip)]
            for k in range(f_skip*8):
                if i + k >= n_frames:
                    x_batch[k % f_skip].append(np.zeros_like(video_cropped[-1]))
                else:
                    x_batch[k % f_skip].append(video_cropped[i+k])
            x_batch = [np.stack(x, axis=0) for x in x_batch]

            # runs the network to get segmentations
            pred, seg_out = sess.run([capsnet.digit_preds, capsnet.segment_layer_sig], 
                                     feed_dict={capsnet.x_input: x_batch,
                                     capsnet.is_train: False,
                                     capsnet.y_input: np.ones((f_skip,), np.int32)*-1})
            '''
            pred = sess.run(capsnet.digit_preds, feed_dict={capsnet.x_input: x_batch,
                                                                 capsnet.is_train: False,
                                                                 capsnet.y_input: np.ones((f_skip,), np.int32)*-1})
            '''
            print('(Batch) Predictions for', name)
            norm_mean = np.mean(pred, axis=0)
            batch_pred_arg = np.argmax(norm_mean)
            print(labels[batch_pred_arg])
            pred.append(norm_mean)
            
            # collects the segmented frames into the correct order
            for k in range(f_skip * 8):
                if i + k >= n_frames:
                    continue

                segmentation_output[i+k] = seg_out[k % f_skip][k//f_skip]

        pred_mean = np.mean(np.stack(pred, axis=0), axis=0)
        video_label = labels[np.argmax(pred_mean)]
        print('Video Prediction for', name + ':', video_label)
        
        # Final segmentation output
        segmentation_output = (segmentation_output >= 0.5).astype(np.int32)

        # Highlights the video based on the segmentation
        alpha = 0.5
        color = np.zeros((3,)) + [0.0, 0, 1.0]
        masked_vid = np.where(np.tile(segmentation_output, [1, 1, 3]) == 1,
                              video_cropped * (1 - alpha) + alpha * color, video_cropped)

        print('Saving Segmented Video')
        vwrite(output_dir+name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))


# inference('../../data/UCF-101/Biking/v_Biking_g01_c03.avi')
# inference('/mnt/data/Rohit/VideoCapsNet/code/SynthVideo/MayurTest2/7200_tr_b_t_MZ.52 831.avi')
# inference('/mnt/data/Rohit/VideoCapsNet/data/SyntheticVideos/Frames/1/7207_tr_r_l_/', dir=True)


dirs = ['../SynthVideo/MayurTest2/Frames/6/17523_Pan1_/',
'../SynthVideo/MayurTest2/Frames/9/13847_tilt1rev_/',
'../SynthVideo/MayurTest2/Frames/10/7326_zoomout_/',
'../SynthVideo/MayurTest2/Frames/0/9184_tr_l_r_/',
'../SynthVideo/MayurTest2/Frames/8/16312_tilt1_/',]
#'''
file_loc = './inference/inputs/'
for vid in os.listdir(file_loc):
    inference(file_loc+vid)
'''
    
for dir in dirs:
    inference(dir, dir=True)
'''
