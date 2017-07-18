# -*- coding: utf-8 -*-
"""
Implement logistic regression to analyze Hill_Valley data
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\lancerts\\Dropbox\\statistics courses\\Applied Machine Learning\\HW\\HW3\\hill-valley")
#import data
X_train = pd.read_csv("X.dat",sep=" ",header=None).as_matrix()
Y_train=  pd.read_csv("Y.dat",sep=" ",header=None).as_matrix().flatten()
X_test=  pd.read_csv("Xtest.dat",sep=" ",header=None).as_matrix()
Y_test=  pd.read_csv("Ytest.dat",sep=" ",header=None).as_matrix().flatten()
X_train.shape
Y_train.shape

#X_train.plot.bar() #EDA of data


'''
produces probablistic estimate for P(y_i = 1 | x_i, w) estimate ranges between 0 and 1.. Feature_matrix~ X, coefficients~ w
'''
def predict_probability(feature_matrix, coefficients):
# Take dot product of feature_matrix and coefficients  
    score = np.dot(feature_matrix,coefficients)

# Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1/(1+np.exp(-score))

# return predictions
    return predictions

"""
Here is the derivative of log likelihood:
errors ~ Yi-Yi_hat 
feature ~ Xi
coefficient ~ wi
l2_penalty ~ lambda
"""
def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, feature_is_constant): 

# Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)

# add L2 penalty term for any feature that isn't the intercept.
    if not feature_is_constant: 
    ## YOUR CODE HERE
            derivative = derivative - 2*l2_penalty*coefficient
    
    return derivative



def compute_log_likelihood_with_L2(feature_matrix,Y, coefficients, l2_penalty):
    indicator = (Y==+1)
    scores = np.dot(feature_matrix, coefficients)

    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)

    return lp 

"""
step_size ~ learning rate 'eta'
l2_penalty ~ coefficient 'lambda'
"""
def logistic_regression_with_L2(feature_matrix,Y, initial_coefficients, step_size, l2_penalty, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    neg_log_likelihood_all = []
    for itr in xrange(max_iter):
    # Predict P(y_i = +1|x_i,w) using your predict_probability() function
    ## YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)
    
    # Compute indicator value for (y_i = +1)
        indicator = (Y ==+1)
    
    # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            is_intercept = (j == 0)
        # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
        # Compute the derivative for coefficients[j]. Save it in a variable called derivative
        ## YOUR CODE HERE
            derivative = feature_derivative_with_L2(errors, feature_matrix[:,j], coefficients[j], l2_penalty, is_intercept)
        
        # add the step size times the derivative to the current coefficient
        ## YOUR CODE HERE
            coefficients[j] = coefficients[j] + step_size*derivative
    
    # compute negative log-likelihood
        lp = compute_log_likelihood_with_L2(feature_matrix, Y, coefficients, l2_penalty)

        neg_log_likelihood_all.append(-lp)
    return coefficients, neg_log_likelihood_all
    
#plot negative log-likelihood
def make_plot(neg_log_likelihood_all, smoothing_window=10, label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    neg_log_likelihood_all_ma = np.convolve(np.array(neg_log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

    plt.plot(np.array(range(smoothing_window-1, len(neg_log_likelihood_all))),
         neg_log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# iteration number')
    plt.ylabel('Negative log likelihood')
    plt.legend(loc='lower right', prop={'size':14})
 
m = list(X_train.shape)[1]


model_1 = logistic_regression_with_L2(X_train, Y_train, initial_coefficients = np.zeros(m), step_size = 4.2e-13, l2_penalty = 0.1, max_iter =1000)

model_2 = logistic_regression_with_L2(X_train, Y_train, initial_coefficients = np.zeros(m), step_size = 4.5e-13, l2_penalty = 0.01, max_iter =1000)

model_3 = logistic_regression_with_L2(X_train, Y_train, initial_coefficients = np.zeros(m), step_size = 4.8e-13, l2_penalty = 0.001, max_iter =1000)

make_plot(list(model_1)[1],)
make_plot(list(model_2)[1],)
make_plot(list(model_3)[1],)
plt.legend(["lambda = 0.1","lambda = 0.01","lambda = 0.001"])


"""
(b) Calculate the misclassification errors
"""
from sklearn.metrics import roc_auc_score
# model_1
pred1_train=predict_probability(X_train, list(model_1)[0])
pred1_test=predict_probability(X_test, list(model_1)[0])
error_train1=roc_auc_score(Y_train, pred1_train)
error_test1=roc_auc_score(Y_test, pred1_test)

#model_2
pred2_train=predict_probability(X_train, list(model_2)[0])
pred2_test=predict_probability(X_test, list(model_2)[0])
error_train2=roc_auc_score(Y_train, pred2_train)
error_test2=roc_auc_score(Y_test, pred2_test)

#model_3
pred3_train=predict_probability(X_train, list(model_3)[0])
pred3_test=predict_probability(X_test, list(model_3)[0])
error_train3=roc_auc_score(Y_train, pred3_train)
error_test3=roc_auc_score(Y_test, pred3_test)
