#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:15:05 2018

@author: vijaygupta1
"""

import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([1]), requires_grad = True)

for i in range(40):
    y = x*x - 5*x + 10
    
    y.backward()
    x.data -= x.grad.data * 0.1
    x.grad.data.zero_()
    print('x = {}, y = {}, iteration = {}'.format(x.data[0],y.data[0], i))

    