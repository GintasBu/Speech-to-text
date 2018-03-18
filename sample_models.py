from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='SRNN_BN')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, N=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
        N - number of identical connected CNN layers
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    for i in range(N):
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if border_mode == 'same':
            output_length = input_length
        elif border_mode == 'valid':
            output_length = input_length - dilated_filter_size + 1
    
        input_length= (output_length + stride - 1) // stride    
    
    
    return input_length

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn1')(input_data)
    # TODO: Add batch normalization
    simp_rnn = BatchNormalization(name='SRNN_BN1')(simp_rnn)
    for i in range(recur_layers-1):
        # Add more recurrent layers
        simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn{}'.format(i+2))(simp_rnn)
        # TODO: Add batch normalization
        simp_rnn = BatchNormalization(name='SRNN_BN{}'.format(i+2))(simp_rnn)    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(simp_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deeper_cnn_GRU_model(input_dim, filters, kernel_size, conv_stride, 
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir'))(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='SRNN_BN')(bidir_rnn)
    bidir_rnn2 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir2'))(bn_rnn)
    bn_rnn2 = BatchNormalization(name='SRNN_BN2')(bidir_rnn2)    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    bidir_rnn3 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir'))(bn_rnn2)
    bn_rnn3 = BatchNormalization(name='SRNN_BN3')(bidir_rnn3)      
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn3)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, N=1)
    print(model.summary())
    return model

def deeper_cnn_GRU_model_dropout(input_dim, filters, kernel_size, conv_stride, pool_size, pool_stride, 
    conv_border_mode, pool_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir', dropout=0.5, recurrent_dropout=0.1))(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='SRNN_BN')(bidir_rnn)
    bidir_rnn2 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir2', recurrent_dropout=0.1))(bn_rnn)
    bn_rnn2 = BatchNormalization(name='SRNN_BN2')(bidir_rnn2)    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    bidir_rnn3 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir', recurrent_dropout=0.1))(bn_rnn2)
    bn_rnn3 = BatchNormalization(name='SRNN_BN3')(bidir_rnn3)      
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn3)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, N=1)
    print(model.summary())
    return model



def final_model(input_dim=161, filters=200, kernel_size=13, conv_stride=2, conv_border_mode='valid', units=200, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir', dropout=0.5, recurrent_dropout=0.1))(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='SRNN_BN')(bidir_rnn)
    bidir_rnn2 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir2', recurrent_dropout=0.1))(bn_rnn)
    bn_rnn2 = BatchNormalization(name='SRNN_BN2')(bidir_rnn2)    
    # TODO: Add last bidirectional layer
    bidir_rnn3 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir', recurrent_dropout=0.1))(bn_rnn2)
    bn_rnn3 = BatchNormalization(name='SRNN_BN3')(bidir_rnn3)      
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn3)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, N=1)
    print(model.summary())
    return model