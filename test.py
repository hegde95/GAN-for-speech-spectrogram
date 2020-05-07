#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:11:08 2020

@author: vineeth, ashwin, shashank
"""

import librosa
import numpy as np
import imageio
import cv2
import sys
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from os import path
# import warnings, ParameterError
from scipy.io.wavfile import write


path_to_models = "./models"
path_to_audio = "./test_audio/SeenAudio"
path_to_results = "./results/GoodAudio"

def get_spectrogram(name_of_audio):
    y, sr = librosa.load(path.join(path_to_audio,name_of_audio),sr=16000)
    max_len = 257
    D = np.abs(librosa.stft(y,n_fft = 512,hop_length = 256))
    amp_max = np.amax(D)
    x = librosa.amplitude_to_db(D, ref=np.max)
    y_len = x.shape[1]
    x = x[:,y_len//2-max_len//2:y_len//2+(max_len+1)//2]
    imageio.imwrite(path.join(path_to_results,name_of_audio[:-4]+".jpg"), x)
    im = cv2.imread(path.join(path_to_results,name_of_audio[:-4]+".jpg"),-1)
    im_f = np.zeros((260,260))
    im_f[:257,:257] = im[:257,:257]
    im_f = (im_f - 127.5) / 127.5
    return im_f  

def griffinlim(S, n_iter=32, hop_length=None, win_length=None, window='hann',
		center=True, dtype=np.float32, length=None, pad_mode='reflect',
		momentum=0.99, init='random', random_state=None):

    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    if momentum > 1:
        warnings.warn('Griffin-Lim with momentum={} > 1 can be unstable. '
                      'Proceed with caution!'.format(momentum))
    elif momentum < 0:
        raise ParameterError('griffinlim() called with momentum={} < 0'.format(momentum))

    # Infer n_fft from the spectrogram shape
    n_fft = 2 * (S.shape[0] - 1)

    # using complex64 will keep the result to minimal necessary precision
    angles = np.empty(S.shape, dtype=np.complex64)
    if init == 'random':
        # randomly initialize the phase
        angles[:] = np.exp(2j * np.pi * rng.rand(*S.shape))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ParameterError("init={} must either None or 'random'".format(init))

    # And initialize the previous iterate to 0
    rebuilt = 0.

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = librosa.istft(S * angles, hop_length=hop_length, win_length=win_length,
                        window=window, center=center, dtype=dtype, length=length)

        # Rebuild the spectrogram
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window, center=center,
                       pad_mode=pad_mode)

        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    # Return the final phase estimates
    return librosa.istft(S * angles, hop_length=hop_length, win_length=win_length,
                 window=window, center=center, dtype=dtype, length=length)

def spec_to_audio(X_out, name_gen):
    imageio.imwrite(path.join(path_to_results,name_gen+".jpg"), X_out.reshape(260,260))
    im = cv2.imread(path.join(path_to_results,name_gen+".jpg"),-1)
    im = im[:257,:257]
    im = (im*80.0/255.0 ) -80.0
    im = librosa.db_to_amplitude(im)
    y2 = griffinlim(im,hop_length=256)
    write(path.join(path_to_results,name_gen+".wav"), 16000, y2*2)
    
def tester(name_of_audio,name_of_model):
    domain2 = name_of_model.split("_")[0].split("2")[1]
    A_real = get_spectrogram(name_of_audio)
    A_real = np.reshape(A_real,(1,260,260,1))
    cust = {'InstanceNormalization': InstanceNormalization}
    model_AtoB = load_model(path.join(path_to_models,name_of_model), cust)
    B_generated  = model_AtoB.predict(A_real)
    name_gen = name_of_audio[:-4] +"_"+domain2+"_generated"
    spec_to_audio(B_generated[0],name_gen)
    
if __name__ == "__main__":
    name_of_audio = "calm_010.wav"
    name_of_model = "Calm2Fearful_generator.h5"
    tester(name_of_audio,name_of_model)
