import cv2
import numpy as np
import os
import pandas as pd

from PIL import Image, ImageDraw
from skvideo.io import vwrite

base_dir = '../../data/YVT/'
ann_dir = 'annotations/'
frames_dir = 'frames/'
# in_h, in_w = 405, 720
out_h, out_w = 256, 480

def save_masked_video(name, video, mask):
    alpha = 0.5
    color = np.zeros((3,)) + [0.0, 0, 1.0]
    masked_vid = np.where(np.tile(mask, [1, 1, 3]) == 1, video * (1 - alpha) + alpha * color, video)
    print('Writing', name + '_segmented.avi')
    vwrite(name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))
    

def create_mask(pts):
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask = Image.fromarray(mask, 'L')
    draw = ImageDraw.Draw(mask)
    for pt in pts:
        draw.polygon(pt.tolist(), fill=1)
    del draw
    mask = np.asarray(mask).copy()
    return mask

    
def resize_and_pad(im):
    in_h, in_w = im.shape[0], im.shape[1]
    if out_w / out_h > in_w / in_h:
        h, w = out_h, in_w * out_h // in_h
    elif out_w / out_h < in_w / in_h:
        h, w =  in_h * out_w // in_w, out_w

    im = cv2.resize(im, (w, h))
    delta_w = out_w - w
    delta_h = out_h - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return im


def create_char_bbox(bbox_list, word_list):
    char_bbox_list = []
    for bbox, word in zip(bbox_list, word_list):
        # bbox is a 8 length vector. word is a string.
        char_count = len(word)
        if char_count == 1:
            char_bbox_list.append(bbox)
            continue

        tl = bbox[0:2]
        tr = bbox[2:4]
        br = bbox[4:6]
        bl = bbox[6:8]

        top_step = (tr - tl) / char_count
        bottom_step = (br - bl) / char_count

        char_bbox_for_word = []
        for _ in range(char_count):
            tr = tl + top_step
            br = bl + bottom_step

            char_bbox = [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]
            char_bbox_for_word.append(char_bbox)

            tl = tr
            bl = br

        # Remove char bbox for spaces
        for idx in range(char_count):
            if word[char_count-idx-1] == ' ':
                char_bbox_for_word.pop(char_count-idx-1)
        
        char_bbox_list.extend(char_bbox_for_word)
        
    return np.array(char_bbox_list, dtype=np.int32)


def merge_words(cluster_1, cluster_2):
    result = []
    if  cluster_2[0] > cluster_1[0]:
        result.extend(cluster_1[0:2])
    else:
        result.extend(cluster_2[0:2])
        
    if cluster_2[2] > cluster_1[2]:
        result.extend(cluster_2[2:4])
    else:
        result.extend(cluster_1[2:4])
        
    if cluster_2[4] > cluster_1[4]:
        result.extend(cluster_2[4:6])
    else:
        result.extend(cluster_1[4:6])
        
    if  cluster_2[0] > cluster_1[0]:
        result.extend(cluster_1[6:8])
    else:
        result.extend(cluster_2[6:8])
    return np.array(result)


def create_line_bbox(word_bbox):
    # Sorting bbox based on mean of y-coordinates. 
    ind = np.argsort(np.mean(word_bbox[:, 1::2], axis=1))
    word_bbox = word_bbox[ind]
    
    clusters = [word_bbox[0]]
    for w_idx in range(1, word_bbox.shape[0]):
        word_width = np.mean(word_bbox[w_idx][[2,4]]) - np.mean(word_bbox[w_idx][[0,6]])
        added_to_cluster = False
        for c_idx in range(1, min(3, len(clusters))+1):
            cur_cluster = clusters[-c_idx]
            cluster_width = np.mean(cur_cluster[[2,4]]) - np.mean(cur_cluster[[0,6]])
            min_width = min(word_width, cluster_width)
            
            cluster_center_x = np.mean(cur_cluster[0::2])
            word_bbox_center_x = np.mean(word_bbox[w_idx][0::2])
            if word_bbox_center_x < cluster_center_x:
                y_window = cur_cluster[1], cur_cluster[7]
                word_bbox_right_y = np.mean(word_bbox[w_idx][[3,5]])
                word_bbox_right_x = np.mean(word_bbox[w_idx][[2,4]])
                cluster_left_x = np.mean(cur_cluster[[0,6]])
                if y_window[0] <= word_bbox_right_y <= y_window[1] and abs(cluster_left_x - word_bbox_right_x) <= min_width:
                    clusters[-c_idx] = merge_words(cur_cluster, word_bbox[w_idx])
                    added_to_cluster = True
                    break

            elif word_bbox_center_x > cluster_center_x:
                y_window = cur_cluster[3], cur_cluster[5]
                word_bbox_left_y = np.mean(word_bbox[w_idx][[1,7]])
                word_bbox_left_x = np.mean(word_bbox[w_idx][[0,6]])
                cluster_right_x = np.mean(cur_cluster[[2,4]])
                if y_window[0] <= word_bbox_left_y <= y_window[1] and abs(word_bbox_left_x - cluster_right_x) <= min_width:
                    clusters[-c_idx] = merge_words(cur_cluster, word_bbox[w_idx])
                    added_to_cluster = True
                    break

            else:
                y_window = np.mean(cur_cluster[[1,3]]), np.mean(cur_cluster[[5,7]])
                word_bbox_center_y = np.mean(word_bbox[w_idx][1::2])
                if y_window[0] <= word_bbox_center_y <= y_window[1]:
                    clusters[-c_idx] = merge_words(cur_cluster, word_bbox[w_idx])
                    added_to_cluster = True
                    break
        if not added_to_cluster:
            clusters.append(word_bbox[w_idx])
    return np.rint(np.array(clusters)).astype(np.int32)
    
    
def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


def merge_lines(bbox_1, bbox_2):    
    top_bbox = bbox_1 if np.mean(bbox_1[[1,3]]) < np.mean(bbox_2[[1,3]]) else bbox_2
    bottom_bbox = bbox_1 if np.mean(bbox_1[[5,7]]) > np.mean(bbox_2[[5,7]]) else bbox_2
    left_bbox = bbox_1 if np.mean(bbox_1[[0,6]]) < np.mean(bbox_2[[0,6]]) else bbox_2
    right_bbox = bbox_1 if np.mean(bbox_1[[2,4]]) > np.mean(bbox_2[[2,4]]) else bbox_2
    
    tl = get_intersect(top_bbox[[0,1]], top_bbox[[2,3]], left_bbox[[0,1]], left_bbox[[6,7]])
    tr = get_intersect(top_bbox[[0,1]], top_bbox[[2,3]], right_bbox[[2,3]], right_bbox[[4,5]])
    br = get_intersect(bottom_bbox[[4,5]], bottom_bbox[[6,7]], right_bbox[[2,3]], right_bbox[[4,5]])
    bl = get_intersect(bottom_bbox[[4,5]], bottom_bbox[[6,7]], left_bbox[[0,1]], left_bbox[[6,7]])
    result_set1 = [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]
    result_set1 = np.array(result_set1)
    
    result = np.zeros((8,))
    result[0], result[1] = min(bbox_1[0], bbox_2[0]), min(bbox_1[1], bbox_2[1]) 
    result[2], result[3] = max(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3])
    result[4], result[5] = max(bbox_1[4], bbox_2[4]), max(bbox_1[5], bbox_2[5])
    result[6], result[7] = min(bbox_1[6], bbox_2[6]), max(bbox_1[7], bbox_2[7])
    
    result[0], result[1] = np.min((result[0], result_set1[0])), np.min((result[1], result_set1[1]))
    result[2], result[3] = np.max((result[2], result_set1[2])), np.min((result[3], result_set1[3]))
    result[4], result[5] = np.max((result[4], result_set1[4])), np.max((result[5], result_set1[5]))
    result[6], result[7] = np.min((result[6], result_set1[6])), np.max((result[7], result_set1[7]))
    return result
    

def create_para_bbox(line_bbox):
    ind = np.argsort(np.mean(line_bbox[:, 1::2], axis=1))
    line_bbox = line_bbox[ind]
    
    clusters = [line_bbox[0]]
    for l_idx in range(1, line_bbox.shape[0]):
        line_bbox_width = np.mean(line_bbox[l_idx][[2,4]]) - np.mean(line_bbox[l_idx][[0,6]])
        line_bbox_height = np.mean(line_bbox[l_idx][[5,7]]) - np.mean(line_bbox[l_idx][[1,3]])
        line_bbox_top = np.array([np.mean(line_bbox[l_idx][[0,2]]), np.mean(line_bbox[l_idx][[1,3]])])
        
        added_to_cluster = False
        for c_idx in range(1, min(3, len(clusters))+1):
            cur_cluster = clusters[-c_idx]
            # Perpendicular distance between line_top (point) and cluster bottom (line)
            dist = np.linalg.norm(np.cross(cur_cluster[[4,5]]-cur_cluster[[6,7]], cur_cluster[[6,7]]-line_bbox_top)) 
            dist = dist / np.linalg.norm(cur_cluster[[4,5]]-cur_cluster[[6,7]])
            
            if dist <= 0.8 * line_bbox_height:
                cluster_width = np.mean(cur_cluster[[2,4]]) - np.mean(cur_cluster[[0,6]])
                if cluster_width > line_bbox_width:
                    wider_bbox = cur_cluster
                    shorter_bbox = line_bbox[l_idx]
                else:
                    wider_bbox = line_bbox[l_idx]
                    shorter_bbox = cur_cluster

                x_window = np.mean(wider_bbox[[0,6]]), np.mean(wider_bbox[[2,4]]) 
                shorter_bbox_center_x = np.mean(shorter_bbox[0::2])
                if x_window[0] <= shorter_bbox_center_x <= x_window[1]:
                    clusters[-c_idx] = merge_lines(wider_bbox, shorter_bbox)
                    added_to_cluster = True
                    break
              
        if not added_to_cluster:
            clusters.append(line_bbox[l_idx])
    return np.rint(np.array(clusters)).astype(np.int32)


def yvt_parse_ann(file):
    colnames=['track id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'] 
    df = pd.read_csv(file, sep=' ', header=None, names=colnames)
    anns = {}
    anns['para_ann'] = {}
    anns['line_ann'] = {}
    anns['word_ann'] = {}
    anns['char_ann'] = {}
    for frame_num in df['frame'].unique():
        pts = df[(df['frame']==frame_num) & (df['lost']!=1) & (df['occluded']!=1)] \
                [['xmin','ymin', 'xmax', 'ymin', 'xmax','ymax', 'xmin', 'ymax']].to_numpy()
        anns['word_ann'][frame_num] = pts
        if pts.size != 0:
            word_list = df[(df['frame']==frame_num) & (df['lost']!=1) & (df['occluded']!=1)] \
                            ['label'].astype(str).values.tolist()
            anns['char_ann'][frame_num] = create_char_bbox(pts, word_list)
            anns['line_ann'][frame_num] = create_line_bbox(pts)
            anns['para_ann'][frame_num] = create_para_bbox(anns['line_ann'][frame_num])
        else:
            anns['line_ann'][frame_num] = np.array([], dtype=np.int32)
            anns['para_ann'][frame_num] = np.array([], dtype=np.int32)
            anns['char_ann'][frame_num] = np.array([], dtype=np.int32)
            
    return anns

base_dir = '/mnt/data/Rohit/VideoCapsNet/data/YVT/'
ann_dir = 'annotations/'
frames_dir = 'frames/'
for split_type in ['train', 'test']:
    for video_name in os.listdir(base_dir+frames_dir+split_type):
        ann_file = base_dir+ann_dir+split_type+'/'+video_name+'.txt'
        anns = yvt_parse_ann(ann_file)
        video_dir = base_dir+frames_dir+split_type+'/'+video_name
        n_frames = os.listdir(video_dir)
        anns['dataset'] = 'yvt'
        for ann_type in ['para_ann', 'line_ann', 'word_ann', 'char_ann']:
            ann = anns[ann_type]
            video = np.zeros((n_frames, out_h, out_w, 3), dtype=np.uint8)
            mask = np.zeros((n_frames, out_h, out_w, 1), dtype=np.uint8)
            for frame_num in range(n_frames):
                frame_loc = video_dir+'/%d.jpg' % frame_num
                frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
                frame_resized = resize_and_pad(frame)
                video[frame_num] = frame_resized
                
                if frame_num in ann:
                    frame_mask = create_mask(ann[frame_num])
                    mask_resized = resize_and_pad(frame_mask)
                    mask[frame_num] = np.expand_dims(mask_resized, axis=-1)
            save_masked_video('./'+ann_type[:4]+'/'+video_name, video/255., mask)
        
        