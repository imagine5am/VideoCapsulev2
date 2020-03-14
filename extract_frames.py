import cv2
import os

'''
To extract frames from videos
'''

dataset_dir = '../../data/UCF-101'
dest = '../../data/UCF101_Frames'
h_out, w_out = 120, 160

for dir in os.listdir(dataset_dir):
    action_dest = dest+'/'+dir
    if not os.path.isdir(action_dest):
        os.mkdir(action_dest)
    for vid in os.listdir(dataset_dir+'/'+dir):
        if vid.endswith(".avi"):
            frame_dest = action_dest+'/'+vid.replace(".avi", '')
            # print(frame_dest+' '+dataset_dir+'/'+dir+'/'+vid); input()
            if not os.path.isdir(frame_dest):
                os.mkdir(frame_dest)
            cap = cv2.VideoCapture(dataset_dir+'/'+dir+'/'+vid)
            # print(dataset_dir+'/'+dir+'/'+vid); input()
            frame_num = 0
            success = 1
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                print(vid + str(frame.shape)+' '+str(frame_num))
                frame = cv2.resize(frame, (w_out, h_out))
                cv2.imwrite(frame_dest+'/'+'frame_'+str(frame_num)+'.jpg', frame)
                frame_num += 1
            cap.release()
