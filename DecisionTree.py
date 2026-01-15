""" 
Decision Tree is a non parametric supervised learning algorithm.
It predicts the value of the target variable by learning 'decision rules'.

Nodes -> Questions on features.
Branches -> Outcomes of the questions.

leaf Nodes -> Final decision or prediction.

Decision treees can be used for both classification and regression tasks.
When to use?

- where interpretibility is important.
- where standard scaling of data is not required.
- No use of Standardization or Normalization.

It measures impurities.
types of impurities:
- Gini Impurity
--> Gini = 1 - sum(p_i^2)
- Entropy(Information Gain)
--> Entropy = - sum(p_i * log2(p_i))

"""
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error