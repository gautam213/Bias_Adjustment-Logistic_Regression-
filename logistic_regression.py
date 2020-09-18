
import numpy as np
from math import *

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logistic_regression(train_X, labels, alpha):
    ''' train_X is in shape of no_samples*no_features
        labels = class of each training samples (no_smaples,1)
        alpha=learning rate
    '''
    n = train_X.shape[1]
    one_column = np.ones((train_X.shape[0],1)) # for bias terms of logistic function
    train_X = np.concatenate((one_column, train_X), axis = 1)
    theta = np.zeros(n+1)
    prediction = logistic_function(theta, train_X, n)
    theta, cost = Gradient_Descent(theta, alpha
                                 , 100000, prediction, train_X, labels, n)
    return theta, cost

def Gradient_Descent(theta, alpha, num_iters, prediction, train_X, labels, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/train_X.shape[0]) * sum(prediction - labels)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/train_X.shape[0]) * np.dot(train_X.transpose()[j],
                                    (prediction - labels))
        prediction = logistic_function(theta, train_X, n)
        cost[i] = (-1/train_X.shape[0]) * sum(labels * np.log(prediction) + (1 - labels) * 
                                        np.log(1 - prediction))

    theta = theta.reshape(1,n+1)
    
    
    return theta, cost
def logistic_function(theta, train_X, n):
    theta=theta.reshape(n+1,1)
    z=np.dot(train_X,theta)
    prediction=sigmoid(z)
    return prediction

def classify(features,theta,threshold):
    one_column = np.ones((features.shape[0],1))
    features = np.concatenate((one_column, features), axis = 1)
    theta=theta.reshape(theta.shape[1],1)
    z=np.dot(features,theta)
    prediction=sigmoid(z)   
    classification=np.array([1 if predict_i >= threshold else 0 for predict_i in prediction])
    return classification
