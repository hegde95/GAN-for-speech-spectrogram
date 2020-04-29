# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:04:05 2020

@author: AshwinTR
"""

import numpy as np
import pandas as pd
from PIL import Image

def load_train_data(file):
    listA = list()
    listB = list()
    data = pd.read_csv(file)
    #print(len(data))
    for i in range(len(data)):
        if data['emotion'][i] == 4: ####sad
            img = np.array((data['pixels'][i].split(' ')), dtype=np.uint8)
            img = np.reshape(img, (48,48,1))
            listA.append(img)
        elif data['emotion'][i] == 3: ####happy
            img = np.array((data['pixels'][i].split(' ')), dtype=np.uint8)
            img = np.reshape(img, (48,48,1))
            listB.append(img)
    #print(np.shape(listA))
    #print(np.shape(listB))
    
    return listA, listB


if __name__=='__main__':
    
# load train_data
    trainA, trainB = load_train_data('D:/USC studies/EE599/project/face_data/train.csv')
    print('trainA', np.shape(trainA))
    print('trainB', np.shape(trainB))
    #im = Image.fromarray(trainB[0].reshape(48,48))
    #im.show()
    filename = 'happy2sad.npz'
    np.savez_compressed(filename, trainA, trainB)
    

