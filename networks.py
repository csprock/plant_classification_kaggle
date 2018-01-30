
# Net_1 (best so far obn 224x224)

def Net_1(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(16, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X_input)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(32, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)    
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)
    X = Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(X)   
    
    X = GlobalMaxPooling2D()(X)
    
    #X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(12, activation = 'softmax', name = 'fc')(X)
    
    the_model = Model(inputs = X_input, outputs = X, name = 'layer_11_model')
    return the_model



