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
1. Bootstrap Aggregating (Bagging): Random Forest uses a technique called bagging.

Situation : How we assume that all decision trees are independent of each other?
Solution : To solve this we use two types 

"""