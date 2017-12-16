import os
from PIL import Image, ImageOps
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import random



def load_and_resize_image(filename, dims = None, rotate = None):
    img = Image.open(filename)
    
    if dims != None:
        img = img.resize(dims, Image.BICUBIC).convert('RGB')
    
    img.load()

    return img


########## import and augment image set ###########
    

random.seed(123)

d = 64
X = np.empty(shape = (0, d,d, 3))
Y = []
augment = True

for c in os.listdir('./train'):
    
    # get list of file names and number of observations in class c
    file_names = os.listdir('./train/' + c)
    n_obs = len(file_names)
    
    
    ims = []
    for f_name in file_names:
        img_main = load_and_resize_image('./train/' + c + '/' + f_name, (d, d))
        
        if augment == True:
            img_aug = ImageOps.mirror(img_main).rotate(random.choice([90, -90, 180, 0]))
            img_aug = np.asarray(img_aug, dtype = 'float32') / 255
            ims.append(img_aug)
            
        img_main = np.asarray(img_main, dtype = 'float32') / 255
        ims.append(img_main)
            

    X_temp = np.stack(ims, axis = 0)
    
    if augment == True:
        Y_temp = (2*n_obs)*[c]
    else:
        Y_temp = n_obs*[c]
    
    X = np.concatenate([X, X_temp])
    Y.extend(Y_temp)


######## create training and validation sets ############

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Yd = np_utils.to_categorical(encoded_Y)

sss = StratifiedShuffleSplit(n_splits = 2, train_size = 0.97, test_size = 0.03, random_state = 123)

for train_index, test_index in sss.split(X,Yd):
    X_train, X_valid = X[train_index,:,:,:], X[test_index,:,:,:]
    Y_train, Y_valid = Yd[train_index,:], Yd[test_index,:]

del X, X_temp, Y, Yd, Y_temp


####### import test set ########

test_set_names = os.listdir('./test/')

X_test = []
for n in test_set_names:
    temp = load_and_resize_image('./test/' + n, (d,d))
    temp = np.asarray(temp, dtype = 'float32') / 255
    X_test.append(temp)
    
X_test = np.stack(X_test, axis = 0)




