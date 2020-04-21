import numpy as np
import librosa
import librosa.display
from scipy.signal import istft
import matplotlib.pyplot as plt
import imageio
import cv2
from scipy.io.wavfile import write

y, sr = librosa.load("orig.wav",sr=16000)
D = np.abs(librosa.stft(y))
amp_max = np.amax(D)
x1 = librosa.amplitude_to_db(D, ref=np.max)
max_x1 = np.amax(x1)
min_x1 = np.amin(x1)
diff = max_x1 - min_x1
imageio.imwrite('spec.jpg', x1)
im = cv2.imread("spec.jpg",0)
im = (im*diff/255.0 ) + min_x1
im = librosa.db_to_amplitude(im,ref=amp_max)
y2 = librosa.griffinlim(im)
write("test.wav", 16000, y2)