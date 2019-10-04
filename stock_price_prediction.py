#!/usr/bin/env python
# coding: utf-8

# In[3]:


import yfinance as yf
# downloading apple stock data from 1st jan 2000 to 1st aug 2019
data = yf.download('AAPL', '2000-01-01', '2019-08-01')


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[5]:


#closing price used for training and testing 
trainingSet = data.Close[:4000]
trainingSet = trainingSet.to_frame()


# In[6]:


testSet = data.Close[4000:]
testSet = testSet.to_frame()


# In[17]:


#normalization to 0 to 1 range
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range = (0, 1))
trainingSetScaled = mms.fit_transform(trainingSet)
testSetScaled = mms.fit_transform(testSet)


# In[18]:


X_train = []
Y_train = []
#appending 40 timestamps
for i in range(40, 4000):
    X_train.append(trainingSetScaled[i-40:i, 0])
    Y_train.append(trainingSetScaled[i, 0])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
#conversion to 3D array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))   


# In[19]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, Y_train, epochs = 100, batch_size = 32)


# In[54]:


#test set pre-processing
datasetFull = pd.concat((trainingSet, testSet), axis = 0)
interim = datasetFull[len(datasetFull) - len(testSet) - 40:].values
interim = interim.reshape(-1,1)
interim = mms.transform(interim)

X_test = []
real_stock_price = []

for i in range(40, 926):
    X_test.append(interim[i-40:i, 0])
    real_stock_price.append(interim[i,0]) 

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

real_stock_price = np.array(real_stock_price)
real_stock_price = real_stock_price.reshape(-1, 1)
real_stock_price = mms.inverse_transform(real_stock_price)

#making predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = mms.inverse_transform(predicted_stock_price)


# In[7]:


#plotting the results
plt.plot(real_stock_price, color = 'black')
plt.plot(predicted_stock_price, color = 'red')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()


# In[ ]:




