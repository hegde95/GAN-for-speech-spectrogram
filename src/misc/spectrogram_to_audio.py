#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:43:37 2020

@author: shashank
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

def read_audio_from_filename(filename):
    audio, sr = librosa.load(filename,sr=16000)
    return audio, sr

def write_audio_to_file(audio,name):
    write(name, 16000, audio)
    return

def make_spectrogram(audio,sr):
    spec = librosa.stft(audio)

    # librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))
    a = np.abs(spec)**2
    a = a/np.amax(a)
    a = 10*np.log10(a)
    # a = a*255/np.amax(a)
    a = a*255/np.amax(a)
    plt.imshow(a)
    plt.show()
    return spec

def make_audio(spec):
    res = librosa.griffinlim(spec)
    return res

def convert_data():
    wav_filename = "./orig.wav"
    w_wav_filename = "test2.wav"    
    audio, sr = read_audio_from_filename(wav_filename)
    spec = make_spectrogram(audio,sr)
    res = make_audio(spec)
    write_audio_to_file(res, w_wav_filename)
    return

specto = convert_data()
import cv2
im = cv2.imread("spec.png",0)