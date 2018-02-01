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


def getLabels(root_dir):
    classes = os.listdir(root_dir)
    classes.sort()
    labels = []
    
    for c in classes:
        file_names = os.listdir(root_dir + '/' + c)
        n = len(file_names)
        labs = n*[c]
        labels.extend(labs)
       
        
    
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    
    return to_categorical(encoded_Y)


def getTrainImages(root_dir, rescale = 1, d):
    
    X = np.empty(shape = (0, d,d, 3))
    classes = os.listdir(root_dir)
    classes.sort()

    for c in classes:
        # get list of file names and number of observations in class c
        file_names = os.listdir(root_dir + '/' + c)
        n_obs = len(file_names)
        
        ims = []
        for f_name in file_names:
            img_main = load_and_resize_image(root_dir + '/' + c + '/' + f_name, (d, d))
            img_main = np.asarray(img_main, dtype = 'float32') / rescale
            ims.append(img_main)
                
        
        X_temp = np.stack(ims, axis = 0)
        X = np.concatenate([X, X_temp])
        
    return X


def getTestImages(root_dir, rescale = 1):
    
    X = np.empty(shape = (0, d,d, 3))

    # get list of file names and number of observations in class c
    file_names = os.listdir(root_dir)
    
    ims = []
    for f_name in file_names:
        img_main = load_and_resize_image(root_dir + '/' + f_name, (d, d))
        img_main = np.asarray(img_main, dtype = 'float32') / rescale
        ims.append(img_main)
            
    X_temp = np.stack(ims, axis = 0)
    
    X = np.concatenate([X, X_temp])
        
    return X



########## import images ###########
    
seed = 123
random.seed(seed)

d = 224
X = np.empty(shape = (0, d,d, 3))
Y = []

classes = os.listdir('./data/train')
classes.sort()

for c in classes:
    
    # get list of file names and number of observations in class c
    file_names = os.listdir('./data/train/' + c)
    n_obs = len(file_names)
    
    ims = []
    for f_name in file_names:
        img_main = load_and_resize_image('./data/train/' + c + '/' + f_name, (d, d))
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

# get labels in training and validation sets

Y_train_label = [Y[i] for i in train_index]
Y_test_label = [Y[i] for i in test_index]


####### labels of Black-grass, Loose Silky-bent #####

Yb_train = [j for i,j in enumerate(Y_train_label) if j in ['Black-grass', 'Loose Silky-bent']]
Yb_test = [j for i,j in enumerate(Y_test_label) if j in ['Black-grass', 'Loose Silky-bent']]


ib_train = np.array([i for i,j in enumerate(Y_train_label) if j in ['Black-grass','Loose Silky-bent']])
ib_test = np.array([i for i,j in enumerate(Y_test_label) if j in ['Black-grass', 'Loose Silky-bent']])


Xb_train = X_train[ib_train,:,:,:]
Xb_valid = X_valid[ib_test,:,:,:]


binary_encoder = LabelEncoder()
binary_encoder.fit(Yb_train)
Yb_train = binary_encoder.transform(Yb_train)
Yb_valid = binary_encoder.transform(Yb_test)
Yb_train = to_categorical(Yb_train)
Yb_valid = to_categorical(Yb_valid)

del X, X_temp, Y, Yd, Y_temp

####### create image generators #######
train_gen = ImageDataGenerator(rescale = 1/255, rotation_range = 90, horizontal_flip = True)

X_train = X_train / 255
X_valid = X_valid / 255
#val_gen = ImageDataGenerator(rescale = 1/255)




####### import test set ########

test_set_names = os.listdir('./test/')

X_test = []
for n in test_set_names:
    temp = load_and_resize_image('./test/' + n, (d,d))
    temp = np.asarray(temp, dtype = 'float32')
    X_test.append(temp)
    
X_test = np.stack(X_test, axis = 0)



#test_gen = ImageDataGenerator(rescale = 1./255)