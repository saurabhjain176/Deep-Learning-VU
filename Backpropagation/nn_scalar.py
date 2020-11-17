#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:27:43 2020

@author: saurabhjain
"""
import math
import numpy as np
from data import load_synth, load_mnist
import matplotlib.pyplot as plt
from random import normalvariate
import random
from sklearn.metrics import classification_report, confusion_matrix

# random.seed(654)

def compute_total_loss(y_hat, y):
    '''
    Calculates the total loss given true labels and predicated labels
    '''
    loss = 0.0
    # Loop through output layer
    for i in range(len(y_hat)):
        # Calculate the loss
        loss += y[i] * (-math.log(y_hat[i]))
    return loss

def compute_softmax(z):
    '''
    Calculates softmax for given list
    '''
    a = [math.e**x for x in z ]
    b =  sum([math.e**x for x in z])
    c = [x/b for x in a]
    return c

def compute_sigmoid(z):
    '''
    Sigmoid calulation
    '''
    sig = 1 / (1 + math.exp(-z))
    return sig

def compute_sigmoid_for_list(list_z):
    h = [0]*len(list_z)
    for i in range(len(list_z)):
        h[i] = compute_sigmoid(list_z[i])
    return h

def initialize_network(method):
    '''
    Initialization function for wights and biases.
    Method - Method to initialize the weights and biases.
    Possible values - {static: initialization using static weights,
                       random: initialization using random numbers. Max 3, min 0.5,
                       he: "He Initialization", this is named for the first author of He et al., 2015}
    '''
    if(method == 'static'):
        w = [[1., 1., 1.], [-1., -1., -1.]]
        v = [[1., 1.], [-1., -1.],[-1., -1.]]
    elif(method == 'random'):
        w = [[normalvariate(3, 1) for _ in range(3)], [normalvariate(0, 2) for _ in range(3)]]
        v = [[normalvariate(3, 1) for _ in range(2)],
              [normalvariate(0, 2) for _ in range(2)],
              [normalvariate(1, 0.5) for _ in range(2)]]

    elif(method == 'he'):
        w = np.random.uniform(-1, 1, size=(2,3)) * np.sqrt(2./3)
        v = np.random.uniform(-1, 1, size=(3,2)) * np.sqrt(2./2)
        # print(w, v)
    b = [0., 0., 0.] #Bias
    c = [0., 0.]
    return w, b, v, c

def linear_calculation(w, b, inputs, length):
    '''
    This function multiplies input valiable with weight and adds bias to it.
    Length - size of the given layer
    '''
    k = [0.]*length
    for j in range(len(k)):
        for i in range(len(inputs)):
            k[j] += w[i][j] * inputs[i]
        k[j] += b[j]
    return k

def forward_pass(w, b, inputs, v, c, y):
    '''
    This function perfoms forward pass of the network.
    1. Performs linear calulation of weights, inputs, and bias for the 1st layer in the network
    2. Computes sigmoid activation on the linear layer
    3. Performs linear calulation of weights, inputs, and bias for the 2st layer in the network
    '''
    k1 = linear_calculation(w, b, inputs, 3)
    h = compute_sigmoid_for_list(k1) #first layer after sigmoid activation
    k2 = linear_calculation(v, c, h, 2)
    y_hat = compute_softmax(k2)

    return h, y_hat

def compute_softmax_der(y_hat, y):
    '''
    This function computes the gradient for softmax
    '''
    dy = [0.,0.]
    for i in range(len(y_hat)):
        dy[i] = y_hat[i] - y[i]
    return dy

def initialize_network_gradients_scalar():
    '''
    This function initialize all the gradients for backpropagation step
    '''
    dv = [[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]]
    dh = [0.0, 0.0, 0.0]
    dk = [0.0, 0.0, 0.0]
    db = [0.0, 0.0, 0.0]
    dw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    dy = [0.,0.]
    return dv, dh, dk, dw, db, dy

def backpropagate_scalar(inputs, y_hat, y, v, h):
    dv, dh, dk, dw, db, dy = initialize_network_gradients_scalar()
    dy = compute_softmax_der(y_hat, y)

    for i in range(len(h)):
        for j in range(len(dy)):
            dv[i][j] = dy[j] * h[i]  # Compute the derivative of V
            dh[i] += dy[j] * v[i][j]  # Compute the derivative of H

    dc = dy.copy() # Assign dC to the parameters dict

    #backward sigmoid
    for i in range(len(h)):
        dk[i] = dh[i]*h[i]*(1-h[i])

    for j in range(len(dk)):
        for i in range(len(inputs)):
            dw[i][j] = dk[j] * inputs[i] # Compute the derivative of W
        db[j] = dk[j]

    return dy, dv, dh, dk, dw, db, dc

def update_parameters_scalar(v, c, w, b, dv, dc, dw, db, learning_rate: float):

    # Declare n_x, n_y, and n_h
    n_x, n_h, n_y = 2, 3, 2
    # Loop through the output layer
    for i in range(n_y):
        # Loop through the hidden layer
        for j in range(n_h):
            # Update the second layer weights
            v[j][i] -= learning_rate * dv[j][i]  # V
        c[i] -= learning_rate * dc[i]  # c

    # Loop through the hidden layer
    for i in range(n_h):
        # Loop through the input layer
        for j in range(n_x):
            # Update the first layer weights
            w[j][i] -= learning_rate * dw[j][i]  # W
        b[i] -= learning_rate * db[i]  # b

    # Return the updated parameters
    return v, c, w, b

def scalar_model():
    epochs = 50
    (xtrain, ytrain), (xval, yval), random = load_synth()
    epoch_loss = []
    epoch_loss1 = []
    w, b, v, c = initialize_network(method = 'random')
    for i in range(epochs):
        for j in range(len(xtrain)):
            inputs = xtrain[j]
            if ytrain[j] == 1: y = [1, 0]
            else: y = [0, 1]
            h, y_hat = forward_pass(w, b, inputs, v, c, y)
            loss = compute_total_loss(y_hat, y)
            dy, dv, dh, dk, dw, db, dc = backpropagate_scalar(inputs, y_hat, y, v, h)
            v, c, w, b = update_parameters_scalar(v, c, w, b, dv, dc, dw, db, 1e-3)
        epoch_loss.append(loss)
        print('Epoch ', i,',Train Loss: ',loss)
        for j in range(len(xval)):
            inputs = xval[j]
            if yval[j] == 1: y = [1, 0]
            else: y = [0, 1]
            h, y_hat = forward_pass(w, b, inputs, v, c, y)
            loss = compute_total_loss(y_hat, y)
        epoch_loss1.append(loss)
        print('Val Loss: ',loss)

    plt.plot(epoch_loss, label='Training')
    plt.plot(epoch_loss1, label='Validation')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    scalar_model()
