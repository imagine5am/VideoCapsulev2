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
        print('All layers:')
        op = sess.graph.get_operations()
        print([m.values() for m in op][1])
        
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.network_save_dir)
        sec_caps_out = sess.run([sess.graph.get_tensor_by_name('sec_caps')], 
                                         feed_dict={capsnet.x_input: x_batch,
                                         capsnet.is_train: False,
                                         capsnet.y_input: np.array([-1], np.int32)})
        print('sec_caps_out.shape', sec_caps_out.get_shape().as_list())
        print('All layers:')
        op = sess.graph.get_operations()
        print([m.values() for m in op][1])
    
    
if __name__=='__main__':
    clip = preprocess()
    corr([clip])