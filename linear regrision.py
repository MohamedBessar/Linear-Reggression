# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 09:21:13 2021

@author: moham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###########################
###########################

####### Read the data #####################
data = pd.read_csv('ex1data1.txt', header= None)
x = data.iloc[:,0] #Read first column
y = data.iloc[:,1] #Read second column
m = len(y)
data.head()

###########################################
###### Drow data ##########################

plt.scatter(x, y)
plt.xlabel('Population of city in 10.000s')
plt.ylabel('Profit in $10.000s')
plt.show()

###########################################
####### Theta parameter ###################

x = x[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
x = np.hstack((ones,x)) #Add 1 column of ones to x
#print(x)

##########################################
######## Compute the cost func ###########

def costfunc(x, y, theta):
    tmp = np.dot(x,theta) - y
    return np.sum(np.power(tmp,2))/(2*m)
j = costfunc(x, y, theta) 
#print(j)

##########################################
####### The Gradient Descent #############

def grad_descent(x,y,theta,alpha,iterations):
    for i in range(iterations):
        tmp = np.dot(x,theta) - y
        tmp = np.dot(x.T, tmp)
        theta = theta - (alpha/m) * tmp
    return theta
theta = grad_descent(x, y, theta, alpha, iterations)
print(theta)
J = costfunc(x, y, theta)
print(J)
#########################################
######## plot the line ###################

plt.scatter(x[:,1], y)
plt.xlabel('Population of city in 10.000s')
plt.ylabel('Profit in $10.000s')
plt.plot(x[:,1], np.dot(x,theta))
plt.show()