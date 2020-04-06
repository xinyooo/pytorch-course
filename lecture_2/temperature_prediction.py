import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn import preprocessing
import datetime
import warnings
warnings.filterwarnings("ignore")

features = pd.read_csv('temps.csv')
print(features.head())

print('Data dimensions: ', features.shape)

years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(dates[:5])

plt.style.use('fivethirtyeight')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')
plt.tight_layout(pad=2)
plt.show()

features = pd.get_dummies(features)
print(features.head(5))

labels = np.array(features['actual'])
features = features.drop('actual', axis = 1)

features_list = list(features.columns)

features = np.array(features)
print(features.shape)

input_features = preprocessing.StandardScaler().fit_transform(features)
print(input_features[0])

x = torch.tensor(input_features, dtype = float)
y = torch.tensor(labels, dtype = float)

# input 14 dimensions vector and output hidden 128 dimensions vector
weights = torch.randn((14, 128), dtype = float, requires_grad = True)
biases = torch.randn(128, dtype = float, requires_grad = True)
# input hidden 128 dimensions vector and output a value of regression
weights2 = torch.randn((128, 1), dtype = float, requires_grad = True)
biases2 = torch.randn(1, dtype = float, requires_grad = True)

learning_rate = 0.001
losses = []

for i in range(1000):
    # forwarding
    hidden = x.mm(weights) + biases
    hidden = torch.relu(hidden)
    predictions = hidden.mm(weights2) + biases2
    # loss eval
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())
    # print out per 100 epochs
    if i % 100 == 0:
        print('loss: ', loss)
    # back propagation
    loss.backward()
    # update parameters
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)
    # clear grad to zero
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

print(predictions.shape)