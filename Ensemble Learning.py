"""
Hello, We have learnt single trees so far now this time to move forward to ensemble learning.

Ensemble learning is a technique that combines the predictions from 
multiple machine learning algorithms to make more accurate predictions 
than any individual model could achieve alone. The main idea is that
by aggregating the outputs of several models, we can reduce the variance and bias, 
leading to improved performance.


In industry we rarely used the single tree models instead we use ensemble learning techniques.

We are going to learn how to combine multiple of weak models to create a strong model.

We are starting with algorithm called Random Forest.

We can understand it as Suppose we are predicting a stock price wheathers Its go up or fall?

Option A: To ask a financial analyst or some expert (Decision Tree).
Option B: To ask mutiple average analysts (Random Forest).
"""
"This is known as Condorcet's Jury Theorem in probability theory."
 
 # Random forest algorithm is the collection of 100+ decision trees which provide outcome 
 # based on majority voting.
 # Theoratical Foundation of random Forest:
"""
1. 
Bootstrap Aggregating (Bagging): Random Forest uses a technique called bagging.

Situation : How we assume that all decision trees are independent of each other?
Solution : To solve this we use two types 

"""
#Random forest implementation in python.
import pandas as pd
import numpy as np
import matplotlib.pyplott as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv('data.csv')
# Preprocess data
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
rf_model.fit(X_train, y_train)
# Make predictions
y_pred = rf_model.predict(X_test)
# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
"""
    of randomness:
    a. Bootstrap Sampling: Each tree is trained on a random sample of the data with replacement.
    b. Feature Randomness: At each split in the tree, a random subset of features is considered 
    for splitting.
    
2. Decision Trees: Each model in the ensemble is a decision tree, which is a simple yet powerful
    algorithm for classification and regression tasks.


"""