import config2 as config
import numpy as np
import os
import tensorflow as tf
from caps_network import Caps3d
from cv2 import resize
from scipy.misc import imread
from skvideo.io import vwrite

def get_first_8_frames(video_dir):
    n_frames = len(os.listdir(video_dir))
    h, w, ch = (128, 240, 3)
    video = np.zeros((n_frames, h, w, ch), dtype=np.uint8)
    for idx in range(n_frames):
        frame = imread(video_dir + ('frame_%d.jpg' % idx))
        frame = resize(frame, (w, h))
        video[idx] = frame
    return video[:8] / 255.

def inference(batch, gpu_config):
    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.save_file_name)
        
        batch_pred, seg_out = sess.run([capsnet.digit_preds, capsnet.segment_layer_sig], 
                                     feed_dict={capsnet.x_input: batch,
                                     capsnet.is_train: False,
                                     capsnet.y_input: np.ones((batch.shape[0],), np.int32)*-1})
        
        batch_pred_arg = np.argmax(batch_pred, axis=1)
        print('Batch Pred:', batch_pred_arg)
        
        i = 0
        for segmentation_output in seg_out:
            clip = batch[i]
            # Final segmentation output
            segmentation_output = (segmentation_output >= 0.5).astype(np.int32)

            # Highlights the video based on the segmentation
            alpha = 0.5
            color = np.zeros((3,)) + [0.0, 0, 1.0]
            masked_vid = np.where(np.tile(segmentation_output, [1, 1, 3]) == 1,
                                  clip * (1 - alpha) + alpha * color, clip)

            print('Saving Segmented Video')
            vwrite(str(i)+'_segmented.avi', (masked_vid * 255).astype(np.uint8))
            i += 0
        
if __name__=='__main__':
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.visible_device_list= '3'
    gpu_config.gpu_options.allow_growth = True
    frames = get_first_8_frames('../SynthVideo/MayurTest2/Frames/6/17523_Pan1_/')
    print(frames.shape)
    # Batch of one clip i.e. 8 frames
    batch = np.expand_dims(frames, axis=0)
    print(batch.shape)
    inference(batch, gpu_config)