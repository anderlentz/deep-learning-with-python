############################################################################
#   Binary classification: Classify movie reviews as positive or negative
#   Used IMDB dataset from keras
#   This exercise is from the book Deep Learning with Python
#   Now we are trying to deal with the overfitting issue by
#   reducing the hidden layer's size to 4. We compare the validation loss
#   from model with hidden layer's size 4 and 16.
############################################################################

from keras.datasets import imdb
from keras import models
from keras import layers
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

# ----- The model definition: hidden layer's size of 16---------

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))

# the final layer uses sigmoid activation to output a probability
model.add(layers.Dense(1, activation='sigmoid'))

# ----- Compiling the model ------

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',  # it is the best option with models that outputs probabilities
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print('Result hidden layer\'s size of 16: \n', results)

"""
    The model definition:
    We are trying to prevent overfitting reducing the model size. Then we are diminishing the model's capacity.
    So, the layer's size was decreased to 4 

"""
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))

# the final layer uses sigmoid activation to output a probability
model.add(layers.Dense(1, activation='sigmoid'))

# ----- Compiling the model ------
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',  # it is the best option with models that outputs probabilities
              metrics=['acc'])

history_size4 = model.fit(partial_x_train,
                          partial_y_train,
                          epochs=20,
                          batch_size=512,

                          validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)
print('Result hidden layer\'s size of 4: \n', results)

# ------ Plotting the training and validation loss -------
history_dict = history.history
history_dict_size4 = history_size4.history

val_loss_values = history_dict['val_loss']
val_size4_loss_values = history_dict_size4['val_loss']
acc = history_dict['acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, val_size4_loss_values, 'bo', label='Smaller model')
plt.plot(epochs, val_loss_values, 'x', label='Original mdel')
plt.title('Validation loss comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()
