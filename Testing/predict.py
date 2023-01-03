'''
    Author: Neel Rajeshbhai Vora
    Mail: neelrajeshbhai.vora@uta.edu
'''
import os
import numpy as np
import librosa
import pandas as pd
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from scipy import signal
from extract_audio import dechunk
from extract_frame import extract_frame



def predict(clip_file_names, chunk_dest,model_path):
    window_size=int(512)
    wd = signal.windows.hamming(window_size)
    slide_size = int(4)
    overlap = window_size - slide_size
    output_pd=[]
    model=load_model(model_path,compile=False)
    for clip1 in clip_file_names:
        row=[]
        channel1_path=chunk_dest+"channel1/"+clip1
        channel2_path=chunk_dest+"channel2/"+clip1
        row.append(clip1[:-4]+".jpg")
        channel1,sample_rate=librosa.load(channel1_path,sr=None)
        channel2,sample_rate=librosa.load(channel2_path,sr=None)
        
        frequency,time,spectrum1=signal.spectrogram(channel1,nfft=window_size,fs=sample_rate,window=wd,noverlap=overlap,mode='magnitude')
        frequency,time,spectrum2=signal.spectrogram(channel2,nfft=window_size,fs=sample_rate,window=wd,noverlap=overlap,mode='magnitude')
        spectrum1=np.expand_dims(spectrum1,axis=0)
        spectrum2=np.expand_dims(spectrum2,axis=0)
        spectrogram=np.stack((spectrum1,spectrum2),axis=-1)

        # normalize
        min=spectrogram.min()
        max=spectrogram.max()
        spectrogram= (spectrogram-min)/(max-min)
        
        preds = model.predict(spectrogram)[0]
        (startX, startY, endX, endY) = preds
        startX,endX=startX*1440, endX*1440
        startY,endY=startY*1080, endY*1080
        width=endX-startX
        height=endY-startY
        row.extend([startX,startY,width,height])
        
        output_pd.append(row)
    output_pd=pd.DataFrame(output_pd)
    output_pd.to_csv('/home/neel/Acoustic/Acoustics/Testing/datas/result.csv', header=False, index=False)