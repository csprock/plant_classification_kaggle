import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import random

def load_and_resize_image(filename, dims = None, rotate = None):
    img = Image.open(filename)
    
    if dims != None:
        img = img.resize(dims, Image.BICUBIC).convert('RGB')
    
    img.load()

    return img


########## import images ###########
    
seed = 123
random.seed(seed)

d = 128
X = np.empty(shape = (0, d,d, 3))
Y = []

classes = os.listdir('./train')
classes.sort()

for c in classes:
    
    # get list of file names and number of observations in class c
    file_names = os.listdir('./train/' + c)
    n_obs = len(file_names)
    
    ims = []
    for f_name in file_names:
        img_main = load_and_resize_image('./train/' + c + '/' + f_name, (d, d))
        img_main = np.asarray(img_main, dtype = 'float32')
        ims.append(img_main)
            

    X_temp = np.stack(ims, axis = 0)
    Y_temp = n_obs*[c]
    
    X = np.concatenate([X, X_temp])
    Y.extend(Y_temp)


######## create training and validation sets ############

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Yd = to_categorical(encoded_Y)

p = 0.05  # set % of data for validation set

sss = StratifiedShuffleSplit(n_splits = 2, train_size = 1-p, test_size = p, random_state = seed)

for train_index, test_index in sss.split(X,Yd):
    X_train, X_valid = X[train_index,:,:,:], X[test_index,:,:,:]
    Y_train, Y_valid = Yd[train_index,:], Yd[test_index,:]

del X, X_temp, Y, Yd, Y_temp


####### import test set ########

test_set_names = os.listdir('./test/')

X_test = []
for n in test_set_names:
    temp = load_and_resize_image('./test/' + n, (d,d))
    temp = np.asarray(temp, dtype = 'float32')
    X_test.append(temp)
    
X_test = np.stack(X_test, axis = 0)


####### create image generators #######
train_gen = ImageDataGenerator(rescale = 1/255, rotation_range = 90, horizontal_flip = True)
val_gen = ImageDataGenerator(rescale = 1/255, rotation_range = 90, horizontal_flip = True)
#test_gen = ImageDataGenerator(rescale = 1./255)