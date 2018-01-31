import numpy as np
#from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model
#from keras.utils import layer_utils
from keras import optimizers
#import keras.backend as K
import keras



def output_dims(n,f,s = 1,p = 0, same = False):
    if same == True: p = (f - 1)/2
    return {'d':np.floor((n + 2*p - f)/s + 1), 'padding':p}



model = Net_1((d,d,3))
model.summary()


opti = optimizers.RMSprop(lr = 1e-3, epsilon = 1e-6)
model.compile(optimizer = opti, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = 0.005, patience = 9)

batch_size = 64
epoch_steps = np.floor(X_train.shape[0] / batch_size)


model.fit_generator(train_gen.flow(X_train,Y_train, batch_size = batch_size), 
                    steps_per_epoch = epoch_steps, 
                    epochs = 250, 
                    validation_data = (X_valid, Y_valid),
                    callbacks = [early_stopping])


model.save('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/Net_1_normal.h5')





#model = keras.models.load_model('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/model1.h5')
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