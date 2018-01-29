
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model
#from keras.utils import layer_utils
from keras import optimizers
import keras

from keras import applications

base_model = applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (256,256,3))

# freeze layers
for lyr in base_model.layers[:168]:
    lyr.trainable = False
    

X = base_model.output
X = Flatten()(X)
X = Dense(512, activation = 'relu')(X)
X = Dropout(0.5)(X)
X = Dense(256, activation = 'relu')(X)
X = Dropout(0.5)(X)
predictions = Dense(12, activation = 'softmax')(X)






final_model = Model(inputs = base_model.input, outputs = predictions)

opti = optimizers.Adam(lr = 1e-4)
final_model.compile(optimizer = opti, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = 0.01, patience = 8)

batch_size = 64
epoch_steps = np.floor(X_train.shape[0] / batch_size)


final_model.fit_generator(train_gen.flow(X_train,Y_train, batch_size = batch_size), 
                    steps_per_epoch = epoch_steps, 
                    epochs = 250, 
                    validation_data = val_gen.flow(X_valid, Y_valid, batch_size = batch_size), 
                    validation_steps = X_valid.shape[0] // batch_size,
                    callbacks = [early_stopping])