
##### 11 layer network #####

def layer_11(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X_input)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(12, activation = 'softmax', name = 'fc')(X)
    
    the_model = Model(inputs = X_input, outputs = X, name = 'layer_11_model')
    return the_model


##### 13 layer network #####
def layer_13(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X_input)
    X = Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(12, activation = 'softmax', name = 'fc')(X)
    
    the_model = Model(inputs = X_input, outputs = X, name = 'layer_13_model')
    return the_model



##### 16 layer network #####
def layer_16(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X_input)
    X = Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(256, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(12, activation = 'softmax', name = 'fc')(X)
    
    the_model = Model(inputs = X_input, outputs = X, name = 'layer_16_model')
    return the_model





















