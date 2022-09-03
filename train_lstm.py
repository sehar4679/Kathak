# Used to train the LSTM model 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import numpy as np
import cv2


steps = 30                              # this the number of frames that is taken from each video
data_location = 'data/train/'           # this is the location of the folder for the data



# lodaing the train data from the numpy arrays
p1_array = np.load(data_location+'pose_1/pose_1.npy')
p2_array = np.load(data_location+'pose_2/pose_2.npy')
x = np.vstack((p1_array,p2_array))
# print(p1_array.shape)
# print(p2_array.shape)
# print(x.shape)



# generating the y data. This is just numbers representing each pose
p1_y = np.zeros((23,),dtype = int)
p2_y = np.ones((23,),dtype = int)
y = np.hstack((p1_y,p2_y))
# print(y)
y = to_categorical(y)
print(y.shape)



n_timesteps, n_features, n_outputs = x.shape[1], x.shape[2], y.shape[1]
print(n_timesteps,n_features,n_outputs)


model = Sequential()
model.add(LSTM(20, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit network
model.fit(x, y, epochs=15, batch_size=2, verbose=0)
model.save('kathak_model')