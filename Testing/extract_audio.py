'''
    Author: Neel Rajeshbhai Vora
    Mail: neelrajeshbhai.vora@uta.edu
'''

import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
def dechunk(source,destination):
    audio_datasets = source
    audio_file_names = []
    for i in reversed(os.listdir(audio_datasets)):
        audio_file_names.append(i[:-4])
    for audio_name in audio_file_names:
        curr_audio_file_path=audio_datasets+"/"+audio_name+".wav"
        curr_audio_file, current_sample_rate = librosa.load(curr_audio_file_path,sr=None,mono=False)
        print(curr_audio_file.shape,current_sample_rate)

        # Divide current audio file into chunks of 0.03 sec audio clip. i.e 15 frames per sec. 48000/30=1600

        upper_limit=1600
        lower_limit=0
        cnt=0
        micro_cnt=1
        clip_audio_datasets= destination
        clip_audio_file_name = audio_name
        while lower_limit < len(curr_audio_file[0]):
            if upper_limit<len(curr_audio_file[0]):
                sf.write(clip_audio_datasets+"channel1/"+clip_audio_file_name+str(cnt)+'_'+str(micro_cnt)+'.wav', curr_audio_file[0][lower_limit:upper_limit], current_sample_rate)
                sf.write(clip_audio_datasets+"channel2/"+clip_audio_file_name+str(cnt)+'_'+str(micro_cnt)+'.wav', curr_audio_file[1][lower_limit:upper_limit], current_sample_rate)
            else:
                sf.write(clip_audio_datasets+"channel1/"+clip_audio_file_name+str(cnt)+'_'+str(micro_cnt)+'.wav', curr_audio_file[0][lower_limit:], current_sample_rate)
                sf.write(clip_audio_datasets+"channel2/"+clip_audio_file_name+str(cnt)+'_'+str(micro_cnt)+'.wav', curr_audio_file[1][lower_limit:], current_sample_rate)
            lower_limit=upper_limit
            micro_cnt+=1
            if micro_cnt>30:
                micro_cnt=1
                cnt+=1
            upper_limit+=1600

# dechunk(source="/home/neel/Acoustic/Acoustics/Testing/datas/audio", destination="/home/neel/Acoustic/Acoustics/Testing/datas/audio_chunks/")  #Source of audio folder | Destination with "/" at end.