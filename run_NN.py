# import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model
from keras.utils import layer_utils
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.utils import plot_model
import keras



def output_dims(n,f,s = 1,p = 0, same = False):
    
    if same == True:
        p = (f - 1)/2
    
    return {'d':np.floor((n + 2*p - f)/s + 1), 'padding':p}



##### define the model ######

def basic_model(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(32, (5,5), strides = (2,2), padding = 'valid')(X_input)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(32, (3,3), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = identity_block(X, 3, [32,32,32], 'A','1')

    X = MaxPooling2D((3,3), strides = (2,2))(X)
    
    X = Conv2D(64, (2,2), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    

    X = identity_block(X, 2, [64,64,64], 'B','1')
    X = identity_block(X, 2, [64,64,64], 'B','2')
    
    X = MaxPooling2D((3,3), strides = (2,2))(X)
    
    X = Conv2D(128, (2,2), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = identity_block(X, 2, [128,128,128], 'C','1')
    X = identity_block(X, 2, [128,128,128], 'C','2')
    #X = identity_block(X, 2, [128,128,128], 'C','3')
    
    X = MaxPooling2D((3,3), strides = (2,2))(X)
    
    X = Conv2D(256, (2,2), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)    
    
    X = identity_block(X, 2, [256,256,256], 'D','1')
    #X = identity_block(X, 2, [256,256,256], 'D','2')

    
    X = GlobalMaxPooling2D()(X)

    #X = Flatten()(X)
    
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(12, activation = 'softmax', name = 'fc')(X)
    
    basic_model = Model(inputs = X_input, outputs = X, name = 'simple model')
    return basic_model


model = basic_model((d,d,3))
model.summary()

train_datagen = ImageDataGenerator()

adam = optimizers.Adam(lr = 1e-4)
model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = 0.01, patience = 12)

batch_size = 32
epoch_steps = np.floor(X_train.shape[0] / 32)

model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size = batch_size), steps_per_epoch = epoch_steps, epochs = 250, validation_data = (X_valid, Y_valid), callbacks = [early_stopping])


model.save('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/model1.h5')

model = keras.models.load_model('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/model1.h5')
#### validation ####
preds = model.evaluate(X_valid, Y_valid)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

##### predict the test data ######

test_predictions = model.predict(X_test)

indices = np.argmax(test_predictions, axis = 1)
labels = []
for i in indices:
    labels.append(classes[i])

output = pd.DataFrame(list(zip(test_set_names, labels)))
output.to_csv('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/test_output.csv')