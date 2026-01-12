"""
Docstring for Linear_Regression2

Manual Implementation for Linear Regression using the interpretation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import  StandardScaler
from sklearn.datasets import fetch_california_housing

class LinearRegression:
    def __init__(self):
        self.theta = None
    
    def fit(self,X,y):
        # 1. Add bias term (column of 1s) to X
        m = X.shape[0]
        X_b = np.c_[np.one((m,1)), X]
        
        # 2. Apply Normal Equation: theta = (X^T * X)^-1 * X^T * y
        X_T_X = X_b.T.dot(X_b)
    
        # Check for singularity (non-invertible matrix)
    
        try:
            self.theta = np.linalg.inv(X_T_X).dot(X_T_X).dot(y)
        except np.linalg.LinAlgError:
            print("Matrix is singular considered regularization (Ridge/Lasso)")
            self.theta = None
    def predict(self, X):
        if self.theta is None: return None
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b.dot(self.theta)
    

        
    