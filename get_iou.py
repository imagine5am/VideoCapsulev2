import numpy as np
import tensorflow as tf
import config2 as config
from caps_network import Caps3d
from load_synth_data import SynthTestDataGenDet as TestDataGen
from tqdm import tqdm

def output_iou(n_vids, n_tot_frames, n_correct, iou_threshs, video_ious, frame_ious):
    print('Label Prediction Accuracy:', n_correct / np.sum(n_vids))
    config.write_output('Label Prediction Accuracy: %.4f\n' % float(n_correct / np.sum(n_vids)))
    for ann_type in config.ann_types:
        print('For ' + ann_type + ':')
        config.write_output('For ' + ann_type + ':\n')
        
        fAP = frame_ious[ann_type] / n_tot_frames
        fmAP = np.mean(fAP, axis=0)
        vAP = video_ious[ann_type] / n_vids
        vmAP = np.mean(vAP, axis=0)
        
        print('IoU, f-mAP:')
        config.write_output('IoU, f-mAP:\n')
        for i in range(20):
            print(str(iou_threshs[i]) + ', ' + str(fmAP[i]))
            config.write_output('%.4f, %.4f\n' % (iou_threshs[i], fmAP[i]))
        config.write_output(str(fAP[:, 10]) + '\n')
        print(fAP[:, 10])   # fAP for IoU = 0.5
        
        print('IoU, v-mAP:')
        config.write_output('IoU, v-mAP:\n')
        for i in range(20):
            print(str(iou_threshs[i]) + ', ' + str(vmAP[i]))
            config.write_output('%.4f, %.4f\n' % (iou_threshs[i], vmAP[i]))
        config.write_output(str(vAP[:, 10]) + '\n')
        print(vAP[:, 10])   # vAP for IoU = 0.5


def iou():
    """
    Calculates the accuracy, f-mAP, and v-mAP over the test set
    """

    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.network_save_dir)

        data_gen = TestDataGen(config.wait_for_data)

        n_correct, n_vids, n_tot_frames = 0, np.zeros((config.n_classes, 1)), np.zeros((config.n_classes, 1))
        iou_threshs = np.arange(0, 20, dtype=np.float32) / 20
        
        frame_ious, video_ious = {}, {}
        for ann_type in config.ann_types: 
            frame_ious[ann_type] = np.zeros((config.n_classes, 20))
            video_ious[ann_type] = np.zeros((config.n_classes, 20))
            

        for _ in tqdm(range(data_gen.n_videos)):
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
                                                                                 capsnet.m: 0.9, capsnet.is_train: False})
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
            n_vids[label] += 1

            for idx, ann_type in enumerate(config.ann_types): 
                pred_segmentations = np.concatenate(segmentations[ann_type], axis=0)
                pred_segmentations = pred_segmentations.reshape((-1, config.vid_h, config.vid_w, 1))
                pred_segmentations = (pred_segmentations >= 0.5).astype(np.int32)

                gt_segmentations = all_gt_segmentations[:, idx, :, :, :]
                seg_plus_gt = pred_segmentations + gt_segmentations

                vid_inter, vid_union = 0, 0
                # calculates f_map
                for i in range(gt_segmentations.shape[0]):
                    frame_gt = gt_segmentations[i]
                    if np.sum(frame_gt) == 0:
                        continue
                    
                    if idx == 0:
                        n_tot_frames[label] += 1

                    inter = np.count_nonzero(seg_plus_gt[i] == 2)
                    union = np.count_nonzero(seg_plus_gt[i])
                    vid_inter += inter
                    vid_union += union

                    i_over_u = inter / union
                    for k in range(iou_threshs.shape[0]):
                        if i_over_u >= iou_threshs[k]:
                            frame_ious[ann_type][label, k] += 1
                
                i_over_u = vid_inter / vid_union
                for k in range(iou_threshs.shape[0]):
                    if i_over_u >= iou_threshs[k]:
                        video_ious[ann_type][label, k] += 1


            if np.sum(n_vids) % 100 == 0:
                print('Finished %d videos' % np.sum(n_vids))
                    
    output_iou(n_vids, n_tot_frames, n_correct, iou_threshs, video_ious, frame_ious)

if __name__=='__main__':
    iou()