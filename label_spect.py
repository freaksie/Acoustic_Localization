'''
    Author: Neel Rajeshbhai Vora
    Mail: neelrajeshbhai.vora@uta.edu
'''

import os
from shutil import move
import librosa
import librosa.display




audio_file_path="/home/wsslab/27_Neel_Vora/Acoustics/dataset/audio/"
audio_files=[]
for i in os.listdir(audio_file_path):
    audio_files.append(i[:-4])
label_file_path="/home/wsslab/27_Neel_Vora/Acoustics/dataset/labels/"

for curr_audio in audio_files:
    curr_audio_path=audio_file_path+curr_audio+".wav"
    file,sample_rate=librosa.load(curr_audio_path,sr=None,mono=False)
    duration=int(len(file[1])/sample_rate)
    print(duration)
    for i in range(duration+1):
        print(i)
        j=1
        while j<31:
            curr_label_path=label_file_path+curr_audio+str(i)+"_"+str(j)+".txt"
            if os.path.isfile(curr_label_path):
                destination="/home/wsslab/27_Neel_Vora/Acoustics/dataset/label-spect/"+curr_audio+str(i)+".txt"
                move(curr_label_path,destination)
                break
            else:
                j+=1
            

        
