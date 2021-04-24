import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
from keras import models, activations, losses, layers

tr = sio.loadmat('data.mat',struct_as_record=True)['train']
ts = sio.loadmat('data.mat',struct_as_record=True)['test']

y_tr = tr[-1,:]
tr = tr[:-1,:]
y_ts = ts[-1,:]
ts = ts[:-1,:]
# ts.reshape((1,ts.shape[0], ts.shape[1]))
# ts.reshape((1,ts.shape[0], ts.shape[1]))
ts = np.expand_dims(ts,axis=0)
tr = np.expand_dims(tr,axis=0)

print(y_tr.shape)
print(tr.shape)

mdl = keras.Sequential()
mdl.add(layers.LSTM(1, activation='linear'))
opt = keras.optimizers.Adadelta(clipvalue=2, learning_rate=0.01)
mdl.build(input_shape=(1, tr.shape[2]))
mdl.compile(optimizer=opt,loss="MeanAbsolutePercentageError",metrics=['MeanAbsolutePercentageError'])
mdl.summary()
mdl.fit(tr,y_tr,batch_size=1,epochs=10, validation_data=(ts,y_ts))
import matplotlib.pyplot as plt
out = mdl.predict(ts[1,:,:])
print(out.shape)
plt.plot(out)
plt.plot(y_ts)
plt.show()
