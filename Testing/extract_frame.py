'''
    Author: Neel Rajeshbhai Vora
    Mail: neelrajeshbhai.vora@uta.edu
'''
import os
import cv2

def extract_frame(source,destination):
    video_data_path=source
    frame_data_path=destination
    video_datas=[]
    for i in os.listdir(video_data_path):
        video_datas.append(i[:-4])

    for curr_video in video_datas:
        curr_path=video_data_path+curr_video+".mp4"
        capture = cv2.VideoCapture(curr_path)
        # print(curr_video+" "+str(capture.get(cv2.CAP_PROP_FPS)))
        frame_num=1
        sec=0
        while(True):
            success, frame = capture.read()
            if success:
                cv2.imwrite(frame_data_path+curr_video+str(sec)+"_"+str(frame_num)+".jpg", frame)
            else:
                break
            if frame_num==30:
                frame_num=1
                sec+=1
            else:
                frame_num+=1
        capture.release()
# extract_frame(,) 
