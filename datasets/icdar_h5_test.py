# import h5py
import cv2
import numpy as np
import os
import skvideo.io  
import pandas as pd
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from skvideo.io import vwrite 

out_h, out_w = 256, 480
train_dict, test_dict = {}, {}


def save_masked_video(name, video, mask):
    alpha = 0.5
    color = np.zeros((3,)) + [0.0, 0, 1.0]
    masked_vid = np.where(np.tile(mask, [1, 1, 3]) == 1, video * (1 - alpha) + alpha * color, video)
    vwrite(name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))
    

def create_mask(shape, pts):
    mask = np.zeros(shape, dtype=np.uint8)
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


# ICDAR
def my_order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    (tl, bl) = leftMost[np.argsort(leftMost[:, 1]), :]
    (tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="int32").flatten()

def order_points(pts):
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "int32")
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.flatten()


def create_char_bbox(bbox, word):
    # bbox is a 8 length vector. word is a string.
    char_count = len(word)
    if char_count==1:
        return [bbox]
    
    tl = np.array([bbox[0], bbox[1]])
    tr = np.array([bbox[2], bbox[3]])
    br = np.array([bbox[4], bbox[5]])
    bl = np.array([bbox[6], bbox[7]])
    
    top_step = (tr - tl) / char_count
    bottom_step = (br - bl) / char_count

    char_bbox_list = []
    for _ in range(char_count):
        tr = tl + top_step
        br = bl + bottom_step
        
        char_bbox = [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]
        char_bbox_list.append(char_bbox)
        
        tl = tr
        bl = br
    
    # Remove char bbox for spaces
    for idx in range(char_count):
        if word[char_count-idx-1] == ' ':
            char_bbox_list.pop(char_count-idx-1)
    
    return char_bbox_list


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
        added_to_cluster = False
        for c_idx in range(1, min(3, len(clusters))+1):
            cur_cluster = clusters[-c_idx]
            cluster_center_x = np.mean(cur_cluster[0::2])
            word_bbox_center_x = np.mean(word_bbox[w_idx][0::2])
            if word_bbox_center_x < cluster_center_x:
                y_window = cur_cluster[1], cur_cluster[7]
                word_bbox_right_y = np.mean(word_bbox[w_idx][[3,5]])
                if y_window[0] <= word_bbox_right_y <= y_window[1]:
                    clusters[-c_idx] = merge_words(cur_cluster, word_bbox[w_idx])
                    added_to_cluster = True
                    break

            elif word_bbox_center_x > cluster_center_x:
                y_window = cur_cluster[3], cur_cluster[5]
                word_bbox_left_y = np.mean(word_bbox[w_idx][[1,7]])
                if y_window[0] <= word_bbox_left_y <= y_window[1]:
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
            
            if dist <= line_bbox_height:
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


def icdar_parse_ann(file):
    '''
    Returns a dict which is something like:
    '''
    anns = {}
    tree = ET.parse(file+'.xml')
    voc = pd.read_csv(file+'.txt', sep=',', header=None, names=['id', 'word'])
    root = tree.getroot()
    anns['para_ann'] = {}
    anns['line_ann'] = {}
    anns['word_ann'] = {}
    anns['char_ann'] = {}
    for frame in root.findall('./frame'):
        frame_num = int(frame.attrib['ID']) - 1
        word_bbox = []
        char_bbox = []

        for object in frame.findall('./object'):
            # Find word bbox
            pts = []
            for pt in object.findall('./Point'):
                pts.append((int(pt.attrib['x']), int(pt.attrib['y'])))
            pts = order_points(np.array(pts))
            word_bbox.append(pts)

            # Find char bbox according to the length of word
            object_id = int(object.attrib['ID'])
            transcription = str(object.attrib['Transcription'])
            if transcription == '##DONT#CARE##' and voc[voc['id']==object_id][['word']].empty:
                char_bbox.append(pts)
            else:
                if transcription != '##DONT#CARE##':
                    word = transcription
                else:
                    word = voc[voc['id']==object_id].iloc[0]['word']
                    # print(word)
                # char_bbox.extend(createCharBB(pts, word))
                char_bbox.extend(create_char_bbox(pts, word))
        
        word_bbox = np.array(word_bbox, dtype=np.int32)
        anns['word_ann'][frame_num] = word_bbox
        anns['char_ann'][frame_num] = np.rint(np.array(char_bbox)).astype(np.int32)
        if word_bbox.size != 0:
            anns['line_ann'][frame_num] = create_line_bbox(word_bbox)
            anns['para_ann'][frame_num] = create_para_bbox(anns['line_ann'][frame_num])
        else:
            anns['line_ann'][frame_num] = word_bbox
            anns['para_ann'][frame_num] = word_bbox
    return anns

base_dirs = {'train': '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_train/',
            'test': '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_test/'
            }
for k, base_dir in base_dirs.items():
    for video_name in [fname for fname in os.listdir(base_dir) if fname.endswith('.mp4')]:
        ann_file = base_dir+video_name[:-4]+'_GT'
        print('Reading', ann_file)
        ann = icdar_parse_ann(ann_file)
        video_loc = base_dir+video_name
        ann['dataset'] = 'icdar'
        if k == 'train':
            train_dict[video_loc] = ann
        else:
            test_dict[video_loc] = ann
        
        video_orig = skvideo.io.vread(video_loc)
        n_frames, h, w, ch = video_orig.shape
        video = np.zeros((n_frames, out_h, out_w, 3), dtype=np.uint8)
        mask = np.zeros((n_frames, out_h, out_w, 1), dtype=np.uint8)
        ann = ann['word_ann']
        for idx in range(n_frames):
            video[idx] = resize_and_pad(video_orig[idx])            
            if idx in ann:
                bbox = ann[idx]
                if bbox.shape != 0:
                    frame_mask = create_mask((h, w), bbox)
                    mask_resized = resize_and_pad(frame_mask)
                    mask[idx] = np.expand_dims(mask_resized, axis=-1)
        save_masked_video('./word/'+video_name[:-4], video/255., mask)
        

'''        
if __name__ == "__main__":
    for video_name in [fname for fname in os.listdir(base_dirs['train']) if fname.endswith('.mp4')]:
        video_orig = skvideo.io.vread(base_dirs['train']+video_name)
        n_frames, h, w, ch = video_orig.shape
        video = np.zeros((n_frames, out_h, out_w, 3), dtype=np.uint8)
        mask = np.zeros((n_frames, out_h, out_w, 1), dtype=np.uint8)
        for idx in range(n_frames):
            video[idx] = resize_and_pad(video_orig[idx])
            ann = train_dict['/home/shivam/CS_Sem2/RnD/VideoCapsulev2/temp/ext_multi_mask/icdar/text_in_Video/ch3_train/Video_2_1_2.mp4']['word_ann']
            if idx in ann:
                bbox = ann[idx]
                if bbox.shape != 0:
                    frame_mask = create_mask((h, w), bbox)
                    mask_resized = resize_and_pad(frame_mask)
                    mask[idx] = np.expand_dims(mask_resized, axis=-1)
        save_masked_video(video_name[:-4], video/255., mask)
'''        