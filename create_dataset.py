'''
    Author: Neel Rajeshbhai Vora
    Mail: neelrajeshbhai.vora@uta.edu
'''

import os
from shutil import move
import librosa
import librosa.display

label_file_path="/home/neel/Acoustic/Acoustics/data/labels/"
labels=[]
for i in os.listdir(label_file_path):
    labels.append(i[:-4])

for label in labels:
    source="/home/neel/Acoustic/Acoustics/m/channel1_"+label+".jpg"
    destination = "/home/neel/Acoustic/Acoustics/data/images/channel1_"+label+".jpg"
    move(source,destination)
    source="/home/neel/Acoustic/Acoustics/m/channel2_"+label+".jpg"
    destination = "/home/neel/Acoustic/Acoustics/data/images/channel2_"+label+".jpg"
    move(source,destination)
