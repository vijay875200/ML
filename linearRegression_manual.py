#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:15:05 2018

@author: vijaygupta1
"""

import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([[1.], [2.], [3.]]))
y = Variable(torch.Tensor([[2.], [4.], [6.]]))

w = Variable(torch.randn(1), requires_grad = True)
#b = Variable(torch.randn(1), requires_grad = True)

for i in range(50):
    yh = x*w 
    loss = ((y - yh)**2).sum()
    loss.backward()
    w.data -= w.grad.data*0.01
#    b.data -= b.grad.data*0.1
    w.grad.data.zero_()
#    b.grad.data.zero_()
    print(w.data[0], loss.data[0], i)
    

    