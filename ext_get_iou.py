import gc
import numpy as np
import tensorflow as tf
import traceback
import ext_config as config
from caps_network import Caps3d
from load_synth_data import SynthTestDataGenDet as TestDataGen
from tqdm import tqdm
from datasets.minetto_gen import Minetto_Gen
from datasets.icdar_gen import ICDAR_Gen
from datasets.yvt_gen import YVT_Gen

def output_iou(n_vids, n_tot_frames, n_correct, iou_threshs, video_ious, frame_ious):
    try:
        output_log = open('minetto_iou.txt', 'w')
        print('Accuracy:', n_correct / n_vids)
        output_log.write('Test Accuracy: %.4f\n' % float(n_correct / n_vids))

        fmAP = frame_ious/n_tot_frames
        vmAP = video_ious/n_vids

        print('IoU f-mAP:')
        output_log.write('IoU f-mAP:\n')
        for i in range(20):
            print('%.2f\t%.2f' % (iou_threshs[i], fmAP[i]))
            output_log.write('%.2f\t%.2f\n' % (iou_threshs[i], fmAP[i]))
        print('IoU v-mAP:')
        output_log.write('IoU v-mAP:\n')
        for i in range(20):
            print('%.2f\t%.2f' % (iou_threshs[i], vmAP[i]))
            output_log.write('%.2f\t%.2f\n' % (iou_threshs[i], vmAP[i]))
            
        output_log.close()
    except:
        print('Unable to save to output log')
        print(traceback.format_exc())

def iou():
    """
    Calculates the accuracy, f-mAP, and v-mAP over the test set
    """

    capsnet = Caps3d()
    with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
        tf.global_variables_initializer().run()
        capsnet.load(sess, config.network_save_dir)

        data_gen = Minetto_Gen()
        # data_gen = ICDAR_Gen()
        # data_gen = YVT_Gen()
        
        # CHANGE 0 to correct label
        n_correct, n_vids, n_tot_frames = 0, 0, 0

        frame_ious = np.zeros((20))
        video_ious = np.zeros((20))
        iou_threshs = np.arange(0, 20, dtype=np.float32)/20

        for _ in tqdm(range(data_gen.n_videos)):
            _, video, bbox = data_gen.get_next_video()
            label = 0
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
                            b_bbox.append(np.zeros((1, config.vid_h, config.vid_w, 1), dtype=np.float32))
                        else:
                            b_vid.append(video[ind:ind+1, :, :, :])
                            b_bbox.append(bbox[ind:ind+1, :, :, :])

                    clips.append((np.concatenate(b_vid, axis=0), np.concatenate(b_bbox, axis=0), label))
                    if np.sum(clips[-1][1]) == 0:
                        clips.pop(-1)

            if len(clips) == 0:
                print('Video has no bounding boxes')
                continue

            batches, gt_segmentations = [], []
            for i in range(0, len(clips), config.batch_size):
                x_batch, bb_batch, y_batch = [], [], []
                for j in range(i, min(i+config.batch_size, len(clips))):
                    x, bb, y = clips[j]
                    x_batch.append(x)
                    bb_batch.append(bb)
                    y_batch.append(y)
                batches.append((x_batch, bb_batch, y_batch))
                gt_segmentations.append(np.stack(bb_batch))

            gt_segmentations = np.concatenate(gt_segmentations, axis=0)
            gt_segmentations = gt_segmentations.reshape((-1, config.vid_h, config.vid_w, 1))  # Shape N_FRAMES, 112, 112, 1

            segmentations, predictions = [], []
            for x_batch, bb_batch, y_batch in batches:
                segmentation, pred = sess.run([capsnet.segment_layer_sig, capsnet.digit_preds],
                                              feed_dict={capsnet.x_input: x_batch, capsnet.y_input: y_batch,
                                                         capsnet.m: 0.9, capsnet.is_train: False})
                segmentations.append(segmentation)
                predictions.append(pred)

            predictions = np.concatenate(predictions, axis=0)
            predictions = predictions.reshape((-1, config.n_classes))
            fin_pred = np.mean(predictions, axis=0)

            fin_pred = np.argmax(fin_pred)
            if fin_pred == label:
                n_correct += 1

            pred_segmentations = np.concatenate(segmentations, axis=0)
            pred_segmentations = pred_segmentations.reshape((-1, config.vid_h, config.vid_w, 1))

            pred_segmentations = (pred_segmentations >= 0.5).astype(np.int32)
            seg_plus_gt = pred_segmentations + gt_segmentations

            vid_inter, vid_union = 0, 0
            # calculates f_map
            for i in range(gt_segmentations.shape[0]):
                frame_gt = gt_segmentations[i]
                if np.sum(frame_gt) == 0:
                    continue

                n_tot_frames += 1

                inter = np.count_nonzero(seg_plus_gt[i] == 2)
                union = np.count_nonzero(seg_plus_gt[i])
                vid_inter += inter
                vid_union += union

                i_over_u = inter / union
                for k in range(iou_threshs.shape[0]):
                    if i_over_u >= iou_threshs[k]:
                        frame_ious[k] += 1

            n_vids += 1
            i_over_u = vid_inter / vid_union
            for k in range(iou_threshs.shape[0]):
                if i_over_u >= iou_threshs[k]:
                    video_ious[k] += 1

            del video, bbox
            gc.collect()
            
            if n_vids % 100 == 0:
                print('Finished %d videos' % n_vids)

        output_iou(n_vids, n_tot_frames, n_correct, iou_threshs, video_ious, frame_ious)
        
iou()