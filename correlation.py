import config2 as config
import cv2
import numpy as np
import os
import tensorflow as tf
from caps_network import Caps3d
from scipy.misc import imread

dataset_dir = '../SynthVideo/out/'

def preprocess():
    video_dir = dataset_dir + "Frames/" + '2/14451_tr_t_b_/'
    n_frames = len(os.listdir(video_dir))
    video = np.zeros((n_frames, config.vid_h, config.vid_w, 3), dtype=np.uint8)
    for idx in range(n_frames):
        frame = imread(video_dir + ('frame_%d.jpg' % idx))
        frame = cv2.resize(frame, (config.vid_w, config.vid_h))
        video[idx] = frame
    clip_len = 8
    start_loc = np.random.randint(0, video.shape[0]-clip_len*2)
    return video[start_loc:start_loc+clip_len*2:2]
    
    
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
                                              capsnet.y_input: np.array([2], np.int32)})
        print('Label:', np.argmax(pred))
        sec_caps = np.mean(sec_caps[0], axis=(-3, -4, -5))
        pred_caps = pred_caps[0][0]
        print('sec_caps.shape', sec_caps.shape)
        print('pred_caps.shape', pred_caps.shape)
        np.save('pred_caps', pred_caps)
        np.save('sec_caps', sec_caps)
    
    
if __name__=='__main__':
    clip = preprocess()
    print('clip.shape', clip.shape)
    corr([clip])
    
'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import linalg as LA
pred_caps = np.load('pred_caps.npy')    # (12, 16)
sec_caps = np.load('sec_caps.npy')[0]   # (32, 16)
corr = np.zeros((12, 32))
for i in range(12):
    for j in range(32):
        corr[i,j] = pred_caps[i] @ sec_caps[j]/(LA.norm(pred_caps[i], 2) * LA.norm(sec_caps[j], 2))

sns.heatmap(corr, annot=True, fmt=".2f", yticklabels=['_tr_l_r_', '_tr_r_l_', '_tr_t_b_', '_tr_b_t_', '_roll_ac_', '_roll_c_', '_Pan1_', '_Panrev1_', '_tilt1_', '_tilt1rev_', '_zoomout_', '_zoomin_',], xticklabels=range(1,33))
plt.title('Correlation between Poses')
plt.xlabel('Secondary Capsules')
plt.ylabel('Prediction Capsules')
plt.show()
'''