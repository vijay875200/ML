#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:15:05 2018

@author: vijaygupta1
"""

import torch
from torch.autograd import Variable
import torch.nn as nn

x = Variable(torch.Tensor([[1.], [2.], [3.]]))
y = Variable(torch.Tensor([[3.], [5.], [7.]]))

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        ypred = self.linear(x)
        return ypred
    
# create an instance of the model
model= Model()

criterion  = nn.MSELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.03)

num_epochs = 200

#training_loop 
for epoch in range(num_epochs):
    yhat = model(x)
    loss = criterion(yhat, y)    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#    print(yhat.data[0][0], yhat.data[1][0], yhat.data[2][0], loss.data[0], epochs
    params = [a for a in model.parameters()]
    print("w is:", params[0].data[0][0], "bias is:", params[1].data[0])

#make a prediction
tinp = Variable(torch.Tensor([[4]]))
tout = model(tinp).data[0][0]
print('test output', tout)