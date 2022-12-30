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
from PIL import Image

def plot_spectrogram(Y, sr, hop_length, img_name, y_axis="linear"):
    plt.figure(figsize=(5,5))
    plt.axis('off')
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.savefig(img_name+".jpg",bbox_inches='tight', pad_inches=0)
    img = Image.open(img_name+".jpg")
    img.thumbnail(size=(300,300))
    print(img)
    img.save(img_name+".jpg", optimize=True, quality=30)



#spectrogram_datasets = "/home/neel/Acoustic/Acoustics/s/"   #Destination location

mel_datasets = "/home/neel/Acoustic/Acoustics/m/"   #Destination location mel spectrogram

curr_frame_size=1024
curr_hop_size=256
clip_audio_datasets = "/home/neel/Acoustic/Acoustics/dataset/datachunks"      #Source Location
clip_audio_file_names = []
for i in (os.listdir(clip_audio_datasets)):
    clip_audio_file_names.append(i[:-4])



# Generate spectrogram

for curr_clip in clip_audio_file_names:
    curr_clip_file_path= clip_audio_datasets+"/"+curr_clip+".wav"
#    destination = spectrogram_datasets+curr_clip
    mel_destination=mel_datasets+curr_clip
    curr_clip_file, curr_sample_rate = librosa.load(curr_clip_file_path, sr=None)
#    spectrogram = librosa.stft(curr_clip_file,n_fft=curr_frame_size,hop_length=curr_hop_size)
    mel_spectrogram=librosa.feature.melspectrogram(curr_clip_file,curr_sample_rate,n_fft=curr_frame_size,hop_length=curr_hop_size,n_mels=80)
    # spectrogram_list contain spectrogram of all the clips in "/home/neel/Acoustic/Acoustics/dataset/datachunks"  Each spectrogram is [1025 X 376] matrix
#    spectrogram=np.abs(spectrogram)**2   #uncomment if you want data with more clear seperation
    mel_spectrogram=np.abs(mel_spectrogram)**2 

    # Save spectrogram as image in destination folder
#    spectrogram_log=librosa.power_to_db(spectrogram)
    mel_spectrogram_log=librosa.power_to_db(mel_spectrogram)
#    plot_spectrogram(spectrogram_log, curr_sample_rate, curr_hop_size, destination,y_axis="log")
 
    plot_spectrogram(mel_spectrogram_log, curr_sample_rate, curr_hop_size, mel_destination,y_axis="mel")
    os.remove(curr_clip_file_path)