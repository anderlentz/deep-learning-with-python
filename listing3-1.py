"""
    Binary classification: Classify movie reviews as positive or negativE
    Used IMDB dataset from keras
    This exercise is from the book Deep Learning with Python, it starts at page 68
"""

from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data,train_labels),(test_data, test_labels) = imdb.load_data(num_words = 10000)

word_index = imdb.get_word_index()

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

decoded_review = ''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

#Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results


#vectorized training and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# ------ Setting aside a validation set ------

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# ----- The model definition ---------

"""
    The argument passed to each Denser layer (16) is the number of hidden units of the layer.
        - hidden unit: is a dimension in the representation space of the layer
        - you can understand the dimensionality of your representation space as "how much freedom you're allowing the
          network to have when learning internal representations"
        - having more hidden units allows your network to learn more complex representations, but it makes the network
          computationally expensive and may lead to learning unwanted patterns  
        
    relu activation implements the following chain of tensor operations:   
        output = relu((doc(W,input)+b)

"""

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))

# the final layer uses sigmoid activation to output a probability
model.add(layers.Dense(1,activation='sigmoid'))


# ----- Compiling the model ------

model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',    #it is the best option with models that outputs probabilities
              metrics=['acc'])



history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data = (x_val,y_val))

results = model.evaluate(x_test, y_test)
print(results)

# ------ Plotting the training and validation loss -------

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']

epochs = range(1,len(acc)+1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the training and validation accuracy, pg 75
plt.clf()
acc_values = history_dict['acc']
val_acc_values= history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




