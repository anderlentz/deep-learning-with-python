'''
    Predict the median price of homes in a given Boston suburb in the mid-1970s
    404 training samples
    102 test samples
    It is a implementation from Deep Learning with Python book, page 85

    Used k-fold cross validation: working with little data, it can help reliably evaluate the model

'''

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Loading the Boston housing dataset
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

# Normalizing the data. The quantities used for nomalizing the test data are computing using training data
mean =  train_data.mean(axis=0) #mean of the feature
train_data -=mean
std = train_data.std(axis=0)
train_data-=std

#Never use in your workflow any quantity computed on the test data, even for a simple nomalization
test_data -=mean    #subtract the mean of the feature
test_data /=std     #and divice by the standard deviation


'''
    This is a simple model definition for scalar regression.
'''
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(128,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(128,activation='relu'))

    # Because the last layer is puraly linear, the net is free to learn to predict values in any range
    model.add(layers.Dense(1))  #single unit and no activation (it will be a linear layer)

    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

'''
    Replace each point with an exponential moving average of the previous points,
    to obtain a smooth curve
'''
def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# K-fold cross validation
k = 4   # number of partitions
num_val_samples = len(train_data) // k  #Data split into k partitions
num_epochs = 100
all_mae_histories = []

for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]     #validation data from partition k
    val_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples]

    # Training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],        #get training data before validation partition i then
         train_data[(i+1)*num_val_samples:]],   #cocatenates with training data after validation partition i
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0
    )

    model = build_model()
    history = model.fit(partial_train_data,
                  partial_train_targets,
                  validation_data=(val_data,val_targets),
                  epochs=num_epochs,
                  batch_size=1,
                  verbose=0)

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)


# Computes the average of the per-epoch MAE scores for all folds
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range (num_epochs)
]

# Excluding the first 10 data points
smoothed_mae_history = smooth_curve(average_mae_history[10:])

print('All mae scores Mean:\n',np.mean(all_mae_histories))

# Plotting validation scores
plt.plot(range(1,len(smoothed_mae_history)+1),smoothed_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()