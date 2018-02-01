import numpy as np
#from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model
#from keras.utils import layer_utils
from keras import optimizers
#import keras.backend as K
import keras

#### train regular model #####

reg_model = Net_1((d,d,3))
reg_model.summary()


opti = optimizers.RMSprop(lr = 1e-3, epsilon = 1e-6)
reg_model.compile(optimizer = opti, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = 0.005, patience = 9)

batch_size = 64
epoch_steps = np.floor(X_train.shape[0] / batch_size)


reg_model.fit_generator(train_gen.flow(X_train,Y_train, batch_size = batch_size), 
                    steps_per_epoch = epoch_steps, 
                    epochs = 250, 
                    validation_data = (X_valid, Y_valid),
                    callbacks = [early_stopping])

reg_model.save('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/Net_1_reg.h5')


#### train binary using transfer learning #####


base_model = Net_1((d,d,3))
base_model.summary()

# initialize base model with weights from regular model
opti = optimizers.Adam(lr = 1e-3, epsilon = 1e-6, decay = 0.0001)
base_model.compile(loss = "categorical_crossentropy", optimizer = opti, metrics = ['accuracy'])
base_model.load_weights('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/Net_1_normal.h5', by_name = True)

# remove first k top layers
k = 6
for i in range(0,k):
    base_model.layers.pop()

#base_model.summary()

# add new layers 
X = base_model.layers[-1].output
X = GlobalMaxPooling2D()(X)
X = Dense(256, activation = 'relu')(X)
X = Dropout(0.5)(X)
X = Dense(128, activation = 'relu')(X)
X = Dropout(0.5)(X)
X = Dense(2, activation = 'softmax')(X)

X_input = base_model.input

binary_model = Model(inputs = X_input, outputs = X)
binary_model.summary()

opti = optimizers.RMSprop(lr = 1e-3, epsilon = 1e-6)

binary_model.compile(loss = "categorical_crossentropy", optimizer = opti, metrics = ['accuracy'])


early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = 0.005, patience = 9)
batch_size = 32
epoch_steps = np.floor(X_train.shape[0] / batch_size)

binary_model.fit_generator(train_gen.flow(Xb_train, Yb_train, batch_size = batch_size), 
                    steps_per_epoch = epoch_steps, 
                    epochs = 250, 
                    validation_data = (Xb_valid, Yb_valid),
                    callbacks = [early_stopping])







