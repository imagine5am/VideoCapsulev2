import gc
import numpy as np
import tensorflow as tf
from caps_network import Caps3d
import ext_config as config
from scipy.misc import imread
from cv2 import resize
import os
from skvideo.io import vwrite
from scipy.misc import imresize
# from datasets.minetto_gen import Minetto_Gen
# from datasets.icdar_gen import ICDAR_Gen
# from datasets.yvt_gen import YVT_Gen
from load_real_data2 import ExternalTestDataLoader
from tqdm import tqdm

'''
def data_gen():
    data_gen = Minetto_Gen()
    # data_gen = ICDAR_Gen()
    # data_gen = YVT_Gen()
    while data_gen.has_data():
        name, video, _ = data_gen.get_next_video()
        yield name, video
'''     

def inference():    
    output_dir = './inference/outputs/'
    capsnet = Caps3d()
    real_data_flag = True if config.data_type=='real' else False
    
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.network_save_dir)
        datasets = ['minetto', 'icdar', 'yvt']
        for dataset_name in datasets:
            data_gen = ExternalTestDataLoader(data_queue_len=6, dataset=dataset_name, sec_to_wait=20)
            for video_idx in tqdm(range(data_gen.n_videos)):
                video, _, _ = data_gen.get_next_video()

                # print('Saving Cropped Video')
                # vwrite(output_dir+name+'_cropped.avi', video)

                n_frames = video.shape[0]

                segmentation_output = {}
                for ann_type in config.ann_types:
                    segmentation_output[ann_type] = np.zeros((n_frames, config.vid_h, config.vid_w, 1))

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
                    batch_pred,  *seg_out = sess.run([capsnet.digit_preds,
                                                      capsnet.segment_layer_sig['para_ann'],
                                                      capsnet.segment_layer_sig['line_ann'],
                                                      capsnet.segment_layer_sig['word_ann'],
                                                      capsnet.segment_layer_sig['char_ann'],
                                                      ], 
                                                     feed_dict={capsnet.x_input: x_batch,
                                                     capsnet.is_train: False, capsnet.is_real: real_data_flag,
                                                     capsnet.y_input: np.ones((f_skip,), np.int32)*-1})

                    norm_mean = np.mean(batch_pred, axis=0)
                    batch_pred_arg = np.argmax(norm_mean)
                    print('(Batch) Predictions for', str(video_idx), 'is', config.labels[batch_pred_arg])
                    pred.append(norm_mean)

                    # collects the segmented frames into the correct order
                    for k in range(f_skip * 8):
                        if i + k >= n_frames:
                            continue
                        for ann_type_itr, ann_type in enumerate(config.ann_types):
                            segmentation_output[ann_type][i+k] = seg_out[ann_type_itr][k % f_skip][k//f_skip]

                pred_mean = np.mean(np.stack(pred, axis=0), axis=0)
                video_label = config.labels[np.argmax(pred_mean)]
                print('Video Prediction for', str(video_idx) + ':', video_label)

                # Final segmentation output
                print('Saving Segmented Video\n')
                for ann_type in config.ann_types:
                    segmentation_output[ann_type] = (segmentation_output[ann_type] >= 0.5).astype(np.int32)

                    # Highlights the video based on the segmentation
                    alpha = 0.5
                    color = np.zeros((3,)) + [0.0, 0, 1.0]
                    masked_vid = np.where(np.tile(segmentation_output[ann_type], [1, 1, 3]) == 1,
                                          video * (1 - alpha) + alpha * color, video)

                    vwrite(output_dir+ann_type[:4]+'/'+str(video_idx)+'_seg.avi', (masked_vid * 255).astype(np.uint8))

                del video, masked_vid
                gc.collect()


if __name__=='__main__':
    inference()
