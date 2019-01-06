from deepConvLSTM import deepConvLSTM
import data_handler as data
import os
from keras.optimizers import Adam

if __name__ == '__main__':
    num_epochs = 1000
    n_classes = 18
    batch_size = 85
    num_features = 113
    timesteps = 24
    rnn_size = 258
    max_len = 150
    learn_rate = 0.001
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    inputs, labels, test_inputs, test_labels = data.load_data()
    shape = inputs.shape[1:]
    model = deepConvLSTM(n_classes, shape)
    opt = Adam(lr=learn_rate)
    model.compile(opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(inputs, labels, batch_size=batch_size,
              epochs=num_epochs, validation_split=0.2)
    loss, acc = model.evaluate(test_inputs, test_labels, steps=1000)
    print("Model accuraccy:", acc, ", loss:", loss)

