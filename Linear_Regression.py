"""
Writing the Linear Regression Core Code. Manual Implementation and Library implementation.

Line Function : y = mx + c.
MSE Error Function : 1/n * {[summation(x-x_mean) * (y - y_mean)]/summation(x-x_mean)}
 In line function : m = Sum(x - x_mean)(y - y_mean)/sum(x - x_mean) ** 2
c = y_mean - m*x_mean
"""
import numpy as np

class ManualLinearRegression:
    def __init__(self):
        self.m = 0
        self.c = 0

    def fit(self, X, y):
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calculate numerator (covariance) and denominator (variance)
        numerator = 0
        denominator = 0
        
        for i in range(len(X)):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2
            
        # Calculate coefficients
        self.m = numerator / denominator
        self.c = y_mean - (self.m * x_mean)
        
    def predict(self, X):
        return self.m * X + self.c

# Test Data (e.g., Rooms Occupied vs Housekeeping Hours)
X_train = np.array([10, 20, 30, 40, 50]) # Input
y_train = np.array([12, 25, 32, 45, 52]) # Target

# Run Manual Model
model = ManualLinearRegression()
model.fit(X_train, y_train)

print(f"Manual Slope (m): {model.m:.2f}")
print(f"Manual Intercept (c): {model.c:.2f}")
print(f"Prediction for 60 rooms: {model.predict(60):.2f} hours")


# Library Function (Scikit-Learn)

import numpy as np
from sklearn.linear_model import LinearRegression

# Reshape X to be a 2D array (required by sklearn)
# -1 means "calculate this dimension automatically" (rows), 1 means 1 column
X_train_reshaped = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
y_train = np.array([12, 25, 32, 45, 52])

# Initialize and Train
sk_model = LinearRegression()
sk_model.fit(X_train_reshaped, y_train)

# Output results
print(f"Library Slope: {sk_model.coef_[0]:.2f}")
print(f"Library Intercept: {sk_model.intercept_:.2f}")

# Predict
prediction = sk_model.predict([[60]])
print(f"Prediction for 60 rooms: {prediction[0]:.2f} hours")
