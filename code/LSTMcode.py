import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
data_training_complete = pd.read_csv('walk_training.csv')  
data_training_processed = data_training_complete.iloc[:, 0:1].values  
from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range = (0, 1))

data_training_scaled = scaler.fit_transform(data_training_processed) 
features_set = []  
labels = []  
for i in range(60, 20000):  
    features_set.append(data_training_scaled[i-60:i, 0])
    labels.append(data_training_scaled[i, 0])
features_set, labels = np.array(features_set), np.array(labels)  
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout  
model = Sequential()  
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  
model.fit(features_set, labels, epochs = 100, batch_size = 32)  

data_testing_complete = pd.read_csv('walk_test.csv')  
data_testing_processed = data_testing_complete.iloc[:, 0:1].values 
"""
data_total = pd.concat((data_training_complete['Open'], data_testing_complete['Open']), axis=0)  
test_inputs = data_total[len(data_total) - len(data_testing_complete) - 60:].values  
test_inputs = test_inputs.reshape(-1,1)  
test_inputs = scaler.transform(test_inputs)  
test_features = []  
for i in range(60, 10000):  
    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)  
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1)) 
"""
data_testing_scaled = scaler.fit_transform(data_testing_processed) 
features_set = []  
labels = []  
for i in range(60, 10000):  
    features_set.append(data_testing_scaled[i-60:i, 0])
    labels.append(data_testing_scaled[i, 0])
features_set, labels = np.array(features_set), np.array(labels)  
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  

predictions = model.predict(features_set)  
predictions = scaler.inverse_transform(predictions)  
plt.figure(figsize=(10,6))  
plt.plot(data_testing_processed, color='blue', label='Actual test data')  
plt.plot(predictions , color='red', label='Predicted data')  
plt.title('data Prediction')  
plt.xlabel('Date')  
plt.ylabel('data')  
plt.legend()  
plt.show()  


