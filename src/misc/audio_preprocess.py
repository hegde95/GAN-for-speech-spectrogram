# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:29:35 2020

@author: AshwinTR
"""

import numpy as np
from PIL import Image
import glob
import h5py

def load_train_data(path1, path2):
    listA = list()
    listB = list()
    #print(len(data))
    for i in glob.glob(path1): ##sad
        img1 = Image.open(i)
        img1 = np.expand_dims(img1, axis=2)
        listA.append(img1)
    #print(len(listA))
    
    for j in glob.glob(path2): ##sad
        img2 = Image.open(j)
        img2 = np.expand_dims(img2, axis=2)
        listB.append(img2)
    
    return listA, listB


if __name__=='__main__':
    
# load train_data
    path1 = 'D:/USC studies/EE599/project/data/image/sad/*'
    path2 = 'D:/USC studies/EE599/project/data/image/happy/*'
    trainA, trainB = load_train_data(path1, path2)
    print('trainA', len(trainA))
    print('trainB', len(trainB))
    
    filename = 'sad2happy_spec.npz'
    np.savez_compressed(filename, trainA, trainB)

# =============================================================================
#     with h5py.File('sad2happy_spec.hdf5', "w") as hf:
#         hf.create_dataset('trainA', data=trainA)
#         hf.create_dataset('trainB', data=trainB)   
#     hf.close()
# =============================================================================
    

