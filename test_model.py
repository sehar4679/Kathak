# this file is used to test the 20 test videos from each pose and
# generate the confidence scores


from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import numpy as np
import cv2




steps = 30                              # this the number of frames that is taken from each video
data_location = 'data/test/'           # this is the location of the folder for the data




# lodaing the train data from the numpy arrays
p1_array = np.load(data_location+'pose_1/pose_1_test_20.npy')
p2_array = np.load(data_location+'pose_2/pose_2_test_20.npy')
x = np.vstack((p1_array,p2_array))
# print(p1_array.shape)
# print(p2_array.shape)
# print(x.shape)




# generating the y data. This is just numbers representing each pose
p1_y = np.zeros((p1_array.shape[0],),dtype = int)
p2_y = np.ones((p2_array.shape[0],),dtype = int)
y = np.hstack((p1_y,p2_y))
print(y)
y = to_categorical(y)
print(y.shape)




n_timesteps, n_features, n_outputs = x.shape[1], x.shape[2], y.shape[1]
print(n_timesteps,n_features,n_outputs)



# loading the saved model back from the disk
model = load_model('kathak_model')


# fit network
# model.fit(x, y, epochs=15, batch_size=4, verbose=0)
# model.save('kathak_model')

_, accuracy = model.evaluate(x, y, batch_size=2, verbose=0)
print(accuracy)

# out = model.predict(np.reshape(x[0],(1,30,50)))
# print(out)

# out = model.predict(x)
# print(out)

# out = model(np.reshape(x[0],(1,30,50)))
# print(out)

out = model.predict(x)
print(out)
