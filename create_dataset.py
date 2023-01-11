'''
    Author: Neel Rajeshbhai Vora
    Mail: neelrajeshbhai.vora@uta.edu
'''

import os
from shutil import move
import librosa
import librosa.display

label_file_path="/home/neel/Acoustic/Acoustics/dataset2/labels/"
labels=[]
for i in os.listdir(label_file_path):
    labels.append(i[:-4])

for label in labels:
    source="/home/neel/Acoustic/Acoustics/dataset2/tmp/channel1_"+label+".wav"
    destination = "/home/neel/Acoustic/Acoustics/dataset2/datachunks/channel1_"+label+".wav"
    move(source,destination)
    source="/home/neel/Acoustic/Acoustics/dataset2/tmp/channel2_"+label+".wav"
    destination = "/home/neel/Acoustic/Acoustics/dataset2/datachunks/channel2_"+label+".wav"
    move(source,destination)