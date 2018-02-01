import numpy as np
import os
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model
from keras import optimizers
import keras
from keras.preprocessing.image import ImageDataGenerator





def output_dims(n,f,s = 1,p = 0, same = False):
    if same == True: p = (f - 1)/2
    return {'d':np.floor((n + 2*p - f)/s + 1), 'padding':p}





d = 64

# define model
model = Net_1((d,d,3))
model.summary()

# compile model
opti = optimizers.RMSprop(lr = 1e-3, epsilon = 1e-6)
model.compile(optimizer = opti, loss='categorical_crossentropy', metrics=['accuracy'])

# define training parameters
early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = 0.005, patience = 9)

batch_size = 64
epoch_steps = np.floor( 4506 / batch_size)

# image generators
train_gen = ImageDataGenerator(rescale = 1/255, rotation_range = 90, horizontal_flip = True, vertical_flip = True)
valid_gen = ImageDataGenerator(rescale = 1/255)

train_generator = train_gen.flow_from_directory(
        './data/train',
        target_size = (d,d),
        batch_size = batch_size,
        class_mode = 'categorical')


valid_gen = valid_gen.flow_from_directory(
        './data/validation',
        target_size = (d,d),
        class_mode = 'categorical')


# train model

model.fit_generator(train_generator, 
                    steps_per_epoch = epoch_steps, 
                    epochs = 250, 
                    validation_data = valid_gen,
                    callbacks = [early_stopping],
                    validation_steps = 244 // batch_size)



# validation
model.evaluate_generator(valid_gen, steps = 10)


##### predict the test data ######

# import test set
classes = os.listdir('./data/train/')
classes.sort()


X_test = getTestImages('./data/test', rescale = 255)

test_predictions = model.predict(X_test)

indices = np.argmax(test_predictions, axis = 1)
labels = []
for i in indices:
    labels.append(classes[i])

output = pd.DataFrame(list(zip(test_set_names, labels)))
output.to_csv('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/test_output.csv')