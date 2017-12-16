

#### create identity block ####


def identity_block(X, f, filters, stage, block):
    
    conv_base_name = 'res' + str(stage) + block
    bn_base_name = 'bn' + str(stage) + block
    
    # output same dimensions as input (except number of channels)
    X_shortcut = X
    
    # first conv layer
    X = Conv2D(filters = filters[0], kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_base_name + '_a')(X)
    X = BatchNormalization(axis = 3, name = bn_base_name + '_a')(X)
    X = Activation('relu')(X)
    
    # second conv layer
    X = Conv2D(filters = filters[1], kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_base_name + '_b')(X)
    X = BatchNormalization(axis = 3, name = bn_base_name + '_b')(X)
    X = Activation('relu')(X)
    
    # third conv layer
    X = Conv2D(filters = filters[2], kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_base_name + '_c')(X)
    X = BatchNormalization(axis = 3, name = bn_base_name + '_c')(X)
    
    
    # add back shortcut 
    X = Add()([X,X_shortcut])
    # apply activation
    X = Activation('relu')(X)
    
    return X

#### create conv block ####
    
def conv_block(X, f, s, filters, stage, block):
    
    conv_base_name = 'res' + str(stage) + block
    bn_base_name = 'bn' + str(stage) + block
    
    X_shortcut = X
    
    # first conv layer
    X = Conv2D(filters = filters[0], kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_base_name + '_a')(X)
    X = BatchNormalization(axis = 3, name = bn_base_name + '_a')(X)
    X = Activation('relu')(X)
    
    # second conv layer
    X = Conv2D(filters = filters[1], kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_base_name + '_b')(X)
    X = BatchNormalization(axis = 3, name = bn_base_name + '_b')(X)
    X = Activation('relu')(X)
    
    # third conv layer
    X = Conv2D(filters = filters[2], kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_base_name + '_c')(X)
    X = BatchNormalization(axis = 3, name = bn_base_name + '_c')(X)
    
    # shortcut conv layer
    X_shortcut = Conv2D(filters = filters[2], kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_base_name + 's')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_base_name + 'c')(X_shortcut)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


'''
CONV2D -> BN -> RELU -> MAXPOOL -> ID_BLOCKx3 -> CONV_BLOCK -> ID_BLOCKx3 -> CONV_BLOCK -> ID_BLOCKx5 -> CONV_BLOCK -> ID_BLOCKx2 -> AVGPOOL -> FCx1
'''



def ResNet50(input_shape, num_classes):
    
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)
    
    ##### Stage 1 #####
    X = Conv2D(64, (7,7), strides = (2,2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_stage_1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides = (2,2))(X)
    
    ##### Stage 2 #####
    F2 = [64,64,256]
    X = conv_block(X, f = 3, s = 2, filters = F2, stage = 2, block = 'A')
    X = identity_block(X, f = 3, filters = F2, stage = 2, block = 'B')
    X = identity_block(X, f = 3, filters = F2, stage = 2, block = 'C')
    
    ##### Stage 3 #####
    F3 = [128,128,512]
    X = conv_block(X, f = 3, s = 2, filters = F3, stage = 3, block = 'A')
    X = identity_block(X, f = 3, filters = F3, stage = 3, block = 'B')
    X = identity_block(X, f = 3, filters = F3, stage = 3, block = 'C')
    X = identity_block(X, f = 3, filters = F3, stage = 3, block = 'D')
    
    ##### Stage 4 #####
    F4 = [256,256,1024]
    X = conv_block(X, f = 3, s = 2, filters = F4, stage = 4, block = 'A')
    X = identity_block(X, f = 3, filters = F4, stage = 4, block = 'B')
    X = identity_block(X, f = 3, filters = F4, stage = 4, block = 'C')
    X = identity_block(X, f = 3, filters = F4, stage = 4, block = 'D')
    X = identity_block(X, f = 3, filters = F4, stage = 4, block = 'E')
    
    ##### Stage 5 #####
    F5 = [512,512,2048]
    X = conv_block(X, f = 3, s = 2, filters = F5, stage = 5, block = 'A')
    X = identity_block(X, f = 3, filters = F5, stage = 5, block = 'B')
    X = identity_block(X, f = 3, filters = F5, stage = 5, block = 'C')
    
    ##### Stage 6 ####
    X = AveragePooling2D(pool_size = (2,2), strides = None, name = 'ave_pool')(X)
    
    X = Flatten()(X)
    X = Dense(num_classes, activation='softmax', name='fc' + str(num_classes))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
































