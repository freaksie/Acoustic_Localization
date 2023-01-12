'''
    Author: Neel Rajeshbhai Vora
    Mail: neelrajeshbhai.vora@uta.edu
'''

import os
from extract_audio import dechunk
from extract_frame import extract_frame
from predict import predict



# --------------------paths-----------------------------------
base_path="/home/neel/Acoustic/Acoustics/Testing/datas"
audio_source=os.path.sep.join([base_path,"audio"])
video_source=os.path.sep.join([base_path,"video/"])
frame_dest=os.path.sep.join([base_path,"frames/"])
frame_output=os.path.sep.join([base_path,"output_frames/"])
chunk_dest=os.path.sep.join([base_path,"audio_chunks/"])
model_path="/home/neel/Acoustic/Acoustics/Testing/models/cnn_dense_5000.h5"



# dechunk(audio_source,chunk_dest)    #destination with / at end
# extract_frame(video_source,frame_dest)   # both with / at end
clip_file_names=[]
for i in os.listdir(chunk_dest+"channel1"):
        clip_file_names.append(i)
predict(clip_file_names,chunk_dest,model_path)



