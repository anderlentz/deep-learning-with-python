'''

    This network classify Reuters newswires into 46 mutually exclusive topics.
    It is a implementation from Deep Learning with Python book, page 78.

'''

from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

#Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

'''
    Vectorization label function
    One hot encoding: used in most cases for categorical data (categorical encoding)
    It embeds each label as an all-zero vector with a 1 in the place of the label index
'''
def to_one_hot(labels,dimension=46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label] = 1.
    return results

#num_words=10000 restricts the data to the 10000 most frequently occurring words found in the data
(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

#vectorize the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#Vectorize the labels, its use from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_one_hot(train_labels) # use categorical(train_labels) from built-in way
on_hot_test_labels = to_one_hot(test_labels)

# Model definition
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))

#for each input sample, the net will output a 46-dimensional vector. Each dimension will encode a different output class
model.add(layers.Dense(46,activation='softmax')) #output a probabily distribution

# Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Setting apart 1000 samples in the training data to validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Training the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))

# Plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the training and validation accuracy

plt.clf() # clears the figure

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs,acc,'bo',label='Traning acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# The network begins to overfit after nine epochs.
plt.show()