import config2 as config
import cv2
import numpy as np
import os
import tensorflow as tf
from caps_network import Caps3d
from scipy.misc import imread

dataset_dir = '../SynthVideo/out/'

def preprocess():
    video_dir = dataset_dir + "Frames/" + '8/16916_tilt1_/'
    n_frames = len(os.listdir(video_dir))
    video = np.zeros((n_frames, config.vid_h, config.vid_w, 3), dtype=np.uint8)
    for idx in range(n_frames):
        frame = imread(video_dir + ('frame_%d.jpg' % idx))
        frame = cv2.resize(frame, (config.vid_w, config.vid_h))
        video[idx] = frame
    clip_len = 8
    start_loc = np.random.randint(0, video.shape[0]-clip_len)
    return video[start_loc:start_loc+clip_len]
    
    
def corr(x_batch):
    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.network_save_dir)
        '''
        print('All layers:')
        for op in tf.get_default_graph().get_operations():
            for t in op.values():
                print(t.name, t.shape)
        '''
        sec_caps, pred_caps, pred = sess.run([capsnet.sec_caps, capsnet.pred_caps, capsnet.digit_preds], 
                                              feed_dict={capsnet.x_input: x_batch,
                                              capsnet.is_train: False,
                                              capsnet.y_input: np.array([-1], np.int32)})
        print('pred.shape: ', pred.shape)
        print('Label:', np.argmax(pred))
        print('sec_caps.shape', sec_caps[0].get_shape())
        print('pred_caps.shape', pred_caps[0].get_shape())
    
    
if __name__=='__main__':
    clip = preprocess()
    corr([clip])