#---------- Convolutional Neural Network for Classification  -------------#
#                                                                         #
# Implementation of Convolutional Neural Network with multiple channels   #
# for classifying MNIST dataset containing hand-written digits (0-9)      #
# using Stochastic Gradient Descent.                                      #
#                                                                         #
# Created by: Vardhan Dongre                                              #
# [ Based on code provided for Logisitic Regression in CS 547 and         #
# the code written for Assignment 1 (Fall 19) ]                           #
#-------------------------------------------------------------------------#

import numpy as np
import h5py
import time
import copy
from random import randint

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
MNIST_data.close()

####################################################################################
#Implementation of stochastic gradient descent algorithm
# Image shape
d, g = 28, 28
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
# Filter size
k_x = 3
k_y = 3
# Number of kernels
num_kernels = 5


kernels = {}
for i in range(num_kernels):
    kernels[i] = np.random.randn(1,k_x,k_y) / np.sqrt(k_x*k_y)
    
k_temp = d-k_y+1
model = {
    'C' : np.random.randn(num_outputs,num_kernels, k_temp, k_temp) / np.sqrt(num_outputs*num_kernels*k_temp**2),
    'b' : np.random.randn(num_outputs,1) / np.sqrt(num_outputs)
}
model_grads = copy.deepcopy(model)

def activation(Z,type = 'ReLU',deri = False):
        # implement the activation function
        if type == 'ReLU':
            if deri == True:
                return 1*(Z>0)
            else:
                return Z*(Z>0)
        elif type == 'Sigmoid':
            if deri == True:
                return 1/(1+np.exp(-Z))*(1-1/(1+np.exp(-Z)))
            else:
                return 1/(1+np.exp(-Z))
        elif type == 'tanh':
            if deri == True:
                return 
            else:
                return 1-(np.tanh(Z))**2
        else:
            raise TypeError('Invalid type!')

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def cross_entropy_error(v,y):
    return -np.log(v[y])


def Conv(x,kernels,model):
    num_kernels = len(kernels)
    x_sp = x.shape
    k_sp = kernels[0].shape
    t_dim = x_sp[1] - k_sp[1] + 1
    result = np.zeros((num_kernels,t_dim,t_dim))
    for i in range(num_kernels):
        for j in range(t_dim):
            for k in range(t_dim):
                result[i,j,k] = np.sum(np.multiply(kernels[i],x[:,j:j+k_sp[1],k:k+k_sp[2]]))
    return result

def forward(x,y,kernels,model):
    X = x.reshape(1,d,d)
    K = kernels
    Z = Conv(X, K, model)
    kdim = d-k_x+1
    H = activation(Z, deri = False).reshape((kdim**2*num_kernels,1))
    U = np.matmul(model['C'].reshape((num_outputs,kdim**2*num_kernels)),H) + model['b']
    predicted = np.squeeze(softmax_function(U))
    p = predicted.reshape((1,num_outputs))
    error = cross_entropy_error(predicted,y)
    results = {
        'Z': Z,
        'H': H,
        'U': U,
        'p':p,
        'error': error
    }
    return results

def backward(x,y,forward_results, kernels, model, model_grads):
    E = np.array([0]*num_outputs).reshape((1,num_outputs))
    E[0][y] = 1
    dU = (-(E - forward_result['p'])).reshape((num_outputs,1))
    model_grads['b'] = copy.copy(dU)
    
    delta = np.zeros((num_kernels, k_temp, k_temp))
    for i in range(10):
        delta += model['C'][i,:]*np.squeeze(dU)[i]
        
    dC = np.zeros((10,num_kernels, k_temp, k_temp))
    for i in range(10):
        dC[i] = np.squeeze(dU)[i]*forward_result['H'].reshape((num_kernels,26,26))
    
    dK = {}
    for i in range(8):
        tdic = {}
        for j in range(1):
            tdic[j] = np.multiply(forward_result['Z'][j],delta[j]).reshape((1,k_temp,k_temp))
        dK[i] = Conv(x.reshape((1,28,28)), tdic, model)
    
    model_grads = {
        'db':model_grads['b'],
        'dC':dC,
        'dK':dK
    }
 
    return model_grads

import time
time1 = time.time()
LR = .01
num_epochs = 100
for epochs in range(num_epochs):
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
        
    total_correct_train = 0
    for n in range(len(x_test)):
        n_random = randint(0,len(x_test)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        forward_result = {}
        forward_result = forward(x, y, kernels, model)
        p_values = forward_result['p']
        prediction = np.argmax(p_values)
        if (prediction == y):
            total_correct_train += 1
        model_grads = backward(x,y,forward_result, kernels, model, model_grads)
        model['C'] -= LR*model_grads['dC']
        model['b'] -= LR*model_grads['db']
        for i in range(num_kernels):
            kernels[i] -= LR*model_grads['dK'][i]
        accuu = total_correct_train/np.float(len(x_test))
    print(accuu)
    
    
time2 = time.time()
print(time2-time1)

######################################################
#test data
total_correct = 0
for n in range(len(x_test)):
    n_random = randint(0, len(x_test)-1)
    y = y_test[n]
    x = x_test[n][:]
    capture = {}
    capture = forward(x, y, kernels, model)
    p = capture['p']
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test))) 