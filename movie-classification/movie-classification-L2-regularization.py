############################################################################
#   Binary classification: Classify movie reviews as positive or negative
#   Used IMDB dataset from keras
#   This exercise is from the book Deep Learning with Python
#   Now we are trying to deal with the overfitting issue by
#   adding L2 weight regularization to the model.
############################################################################

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ''.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# vectorized training and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# ------ Setting aside a validation set ------

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# The model definition: hidden layer's size of 16, without L2 regularization
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',  # it is the best option with models that outputs probabilities
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

results_without_l2_reg = model.evaluate(x_test, y_test)

# The model definition: adding L2 regularization to the previews model
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #sigmoid activation to output a probability

# Compiling the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',  # it is the best option with models that outputs probabilities
              metrics=['acc'])

history_L2_reg = model.fit(partial_x_train,
                           partial_y_train,
                           epochs=20,
                           batch_size=512,
                           validation_data=(x_val, y_val))

results_with_L2 = model.evaluate(x_test, y_test)

# Results
print('Result without L2 regularization: \n Loss: ', results_without_l2_reg[0],' \nAccuracy: ',results_without_l2_reg[1])
print('Result with L2 regularization: \n Loss: ', results_with_L2[0], '\nAccuracy: ',results_with_L2[1])

# Plotting the training and validation loss
history_dict = history.history
history_dict_L2_reg = history_L2_reg.history

val_loss_values = history_dict['val_loss']
val_size_L2_regularized = history_dict_L2_reg['val_loss']
acc = history_dict['acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, val_loss_values, 'x', label='Original model')
plt.plot(epochs, val_size_L2_regularized, 'bo', label='L2-regularized model')
plt.title('Validation loss comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()
