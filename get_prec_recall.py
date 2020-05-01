import numpy as np
import tensorflow as tf
import traceback
import config as config
from caps_network import Caps3d
from load_synth_data import SynthTestDataGenDet as TestDataGen
from load_real_data import ExternalTrainDataLoader, ExternalTestDataLoader
from tqdm import tqdm


def get_precision_recall():
    """
    Calculates the Precision, Recall and F1-measure over the test set
    """

    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.network_save_dir)

        if config.data_type == 'synth':
            data_gen = data_gen = TestDataGen(config.wait_for_data)
        elif config.data_type == 'real':
            data_gen = ExternalTestDataLoader()
                
        tp, fn, fp = {}, {}, {}
        for ann_type in config.ann_types: 
            tp[ann_type] = 0
            fn[ann_type] = 0
            fp[ann_type] = 0
            
        for video_idx in tqdm(range(data_gen.n_videos)):
            video, bbox, label = data_gen.get_next_video()

            f_skip = config.frame_skip
            clips = []
            n_frames = video.shape[0]
            for i in range(0, video.shape[0], 8*f_skip):
                for j in range(f_skip):
                    b_vid, b_bbox = [], []
                    for k in range(8):
                        ind = i + j + k*f_skip
                        if ind >= n_frames:
                            b_vid.append(np.zeros((1, config.vid_h, config.vid_w, 3), dtype=np.float32))
                            b_bbox.append(np.zeros((1, len(config.ann_types), config.vid_h, config.vid_w, 1), dtype=np.float32))
                        else:
                            b_vid.append(video[ind:ind+1, :, :, :])
                            b_bbox.append(bbox[ind:ind+1, :, :, :, :])

                    clips.append((np.concatenate(b_vid, axis=0), np.concatenate(b_bbox, axis=0), label))
                    if np.sum(clips[-1][1]) == 0:
                        clips.pop(-1)

            if len(clips) == 0:
                print('Video has no bounding boxes')
                continue

            batches, all_gt_segmentations = [], []
            for i in range(0, len(clips), config.batch_size):
                x_batch, bb_batch, y_batch = [], [], []
                for j in range(i, min(i+config.batch_size, len(clips))):
                    x, bb, y = clips[j]
                    x_batch.append(x)
                    bb_batch.append(bb)
                    y_batch.append(y)
                batches.append((x_batch, bb_batch, y_batch))
                all_gt_segmentations.append(np.stack(bb_batch))

            all_gt_segmentations = np.concatenate(all_gt_segmentations, axis=0)
            all_gt_segmentations = all_gt_segmentations.reshape((-1, len(config.ann_types), config.vid_h, config.vid_w, 1))  # Shape N_FRAMES, 4,112, 112, 1

            segmentations = {}
            for ann_type in config.ann_types:
                segmentations[ann_type] = []
            predictions = []
            for x_batch, bb_batch, y_batch in batches:
                seg_para, seg_line, seg_word, seg_char, pred = sess.run([capsnet.segment_layer_sig['para_ann'],
                                                                         capsnet.segment_layer_sig['line_ann'],
                                                                         capsnet.segment_layer_sig['word_ann'],
                                                                         capsnet.segment_layer_sig['char_ann'],
                                                                         capsnet.digit_preds],
                                                                      feed_dict={capsnet.x_input: x_batch, 
                                                                                 capsnet.y_input: y_batch,
                                                                                 capsnet.m: 0.9, capsnet.is_train: False,
                                                                                 capsnet.is_real: capsnet.real_data_flag})
                segmentations['para_ann'].append(seg_para)
                segmentations['line_ann'].append(seg_line)
                segmentations['word_ann'].append(seg_word)
                segmentations['char_ann'].append(seg_char)
                predictions.append(pred)

            predictions = np.concatenate(predictions, axis=0)
            predictions = predictions.reshape((-1, config.n_classes))
            fin_pred = np.mean(predictions, axis=0)
            fin_pred = np.argmax(fin_pred)
            if fin_pred == label:
                n_correct += 1

            for idx, ann_type in enumerate(config.ann_types): 
                pred_segmentations = np.concatenate(segmentations[ann_type], axis=0)
                pred_segmentations = pred_segmentations.reshape((-1, config.vid_h, config.vid_w, 1))
                pred_segmentations = (pred_segmentations >= 0.5).astype(np.int32)

                gt_segmentations = all_gt_segmentations[:, idx, :, :, :]
                
                tp[ann_type] += np.count_nonzero(np.logical_and(pred_segmentations==1, gt_segmentations==1))
                fp[ann_type] += np.count_nonzero(np.logical_and(pred_segmentations==1, gt_segmentations==0))
                fn[ann_type] += np.count_nonzero(np.logical_and(pred_segmentations==0, gt_segmentations==1))

            if (video_idx + 1) % 100 == 0:
                print('Finished %d videos' % (video_idx + 1))
                
        for ann_type in enumerate(config.ann_types):
            precision = tp[ann_type]/ (tp[ann_type] + fp[ann_type])
            recall = tp[ann_type]/ (tp[ann_type] + fn[ann_type])
            f1 =  2 * precision * recall / (precision + recall)
            print('For', ann_type+':')
            print('Precision: %.2f, Recall: %.2f, F1: %.2f' % (precision, recall, f1))
                

if __name__=='__main__':
    get_precision_recall()