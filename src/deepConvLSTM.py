from keras.layers import (CuDNNLSTM, Dropout, Dense,
                                     Conv1D, MaxPooling1D, Activation)
from keras import Sequential


class deepConvLSTM(Sequential):
    def __init__(self, num_classes, input_shape, rnn_size=128, num_rnn_layers=2,
                 filter_size=[128, 256, 256], kernel_size=[3,3,3],
                 pool_size=[2,2,2], num_cnn_layers=3, dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.input_data_shape = input_shape
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.dropout_rate = dropout_rate
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_cnn_layers = num_cnn_layers
        self._create_model()

    def _create_model(self):
        self.add(Conv1D(self.filter_size[0], self.kernel_size[0],
                 input_shape=self.input_data_shape))
        self.add(MaxPooling1D(self.pool_size[0]))
        self.add(Activation('relu'))
        for layer in range(1, self.num_cnn_layers):    
            self.add(Conv1D(self.filter_size[layer], self.kernel_size[layer]))
            self.add(MaxPooling1D(self.pool_size[layer]))
            self.add(Activation('relu'))
        self.add(CuDNNLSTM(self.rnn_size,
                 return_sequences=True))
        for layer in range(1, self.num_rnn_layers):
            self.add(CuDNNLSTM(self.rnn_size))
        if self.dropout_rate > 0:
            self.add(Dropout(self.dropout_rate))
        self.add(Dense(self.num_classes, activation='softmax'))
