#!/usr/bin/env python
# coding: utf-8

# In[42]:


import yfinance as yf
# downloading apple stock data from 1st jan 2000 to 1st aug 2019
data = yf.download('AAPL', '2000-01-01', '2019-08-01')


# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[44]:


#closing price used for training and testing 
trainingSet = data.Close[:4000]
trainingSet = trainingSet.to_frame()
#exponential smoothning over 5 days 
trainingSet = trainingSet.ewm(span = 5).mean()


# In[45]:


testSet = data.Close[4000:]
testSet = testSet.to_frame()
#exponential smoothning over 5 days 
testSet = testSet.ewm(span = 5).mean()


# In[46]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range = (0, 1))
trainingSetScaled = mms.fit_transform(trainingSet)
testSetScaled = mms.fit_transform(testSet)


# In[47]:


X_train = []
Y_train = [] 

#learning from past 80 timestamps and predicting for future 20
for i in range(101, 4000):
    X_train.append(trainingSetScaled[i-101:i-20, 0])
    Y_train.append(trainingSetScaled[i-20:i, 0])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))#conversion to 3D array 


# In[48]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()

model.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 70))
model.add(Dropout(0.2))

model.add(Dense(units = 20))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, Y_train, epochs = 100, batch_size = 32)


# In[50]:


#test set pre-processing
dataset_full = pd.concat((trainingSet, testSet), axis = 0)
interim = dataset_full[len(dataset_full) - len(testSet) - 60:].values
interim = interim.reshape(-1,1)
interim = mms.transform(interim)

X_test = []
real_stock_price = []

for i in range(101, 926):
    X_test.append(interim[i-101:i-20, 0])
    real_stock_price.append(interim[i-20:i,0]) 

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

real_stock_price = np.array(real_stock_price)
real_stock_price = real_stock_price.reshape(-1, 20)
real_stock_price = mms.inverse_transform(real_stock_price)

#making predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = mms.inverse_transform(predicted_stock_price)

predicted = []
real = []
for i in range(0, 41):
    predicted = np.append(predicted, predicted_stock_price[20*i,:])
    real = np.append(real, real_stock_price[20*i,:])


# In[52]:


#plotting the results
plt.plot(real, color = 'black')
plt.plot(predicted, color = 'red')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()

