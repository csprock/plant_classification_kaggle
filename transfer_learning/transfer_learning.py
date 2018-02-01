import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model
from keras import optimizers
import keras
from keras import applications





base_model = applications.vgg16.VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))


#datagen = ImageDataGenerator(rotation_range = 90, 
#                              width_shift_range = 0.1, 
#                              height_shift_range = 0.1,
#                              horizontal_flip = True,
#                              vertical_flip = True, 
#                              rescale = 1/255)

#
#train_gen = datagen.flow_from_directory('./data/train', target_size = (224,224), class_mode = 'categorical')
#val_gen = datagen.flow_from_directory('./data/validation', target_size = (224,224), class_mode = 'categorical')
#


temp = base_model.predict(X, verbose = True)



X_input = Input((7,7,512))
X = GlobalMaxPooling2D()(X_input)
X = Dense(512, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(256, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(12, activation = 'softmax')(X)


new_model = Model(inputs = X_input, outputs = X)



#final_model = Model(inputs = base_model.input, outputs = predictions)

opti = optimizers.Adam(lr = 1e-3)
new_model.compile(optimizer = opti, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = 0.01, patience = 8)

batch_size = 128
epoch_steps = np.floor(4506 / batch_size)


new_model.fit(x = temp, y = Y_train, epochs = 250, validation_split = 0.05)


#final_model.fit_generator(train_gen.flow(X_train,Y_train, batch_size = batch_size), 
#                    steps_per_epoch = epoch_steps, 
#                    epochs = 250, 
#                    validation_data = val_gen.flow(X_valid, Y_valid, batch_size = batch_size), 
#                    validation_steps = X_valid.shape[0] // batch_size,
#                    callbacks = [early_stopping])

# import test set
classes = os.listdir('./data/train/')
classes.sort()


X_test = getTestImages('./data/test', rescale = 255)

test_input = base_model.predict(X_test, verbose = True)

test_predictions = new_model.predict(test_input)

indices = np.argmax(test_predictions, axis = 1)
labels = []
for i in indices:
    labels.append(classes[i])

output = pd.DataFrame(list(zip(test_set_names, labels)))
output.to_csv('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/test_output.csv')