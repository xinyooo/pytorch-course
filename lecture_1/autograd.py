import torch
import torch.nn as nn
import numpy as np

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
print(x_train.shape)

y_value = [2 * i + 1 for i in x_values]
y_train = np.array(y_value, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(y_train.shape)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)
print(model)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('This program will use {} to compute.'.format(device))
model.to(device)

epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch += 1
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)
    # clear grad
    optimizer.zero_grad()
    # forwarding
    outputs = model(inputs)
    # eval loss
    loss = criterion(outputs, labels)
    # backwarding
    loss.backward()
    # update weight
    optimizer.step()

    if epoch%50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

predicted = model(torch.from_numpy(x_train).to(device).requires_grad_()).cpu().data.numpy()
print(predicted)

torch.save(model.state_dict(), 'autograd_model.pkl')

print(model.load_state_dict(torch.load('autograd_model.pkl')))