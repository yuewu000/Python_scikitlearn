"""
Created on Tue Jan 07 17:35:11 2017
We design a three-layer neural network model by sklearn
The model is tuned by randomized search on the following parameters:
    learning_rate
    hidden0__units
    hidden0__type
The model is summarized after training.
    
@author: Yue Wu
"""

import numpy as np
from sknn.mlp import Classifier, Layer
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV

# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

n_feat = 800
n_target = 1

# Create a 3-layer neural network
nn = Classifier(
        layers=[
            Layer("Tanh", units=12),
            Layer("Sigmoid", units=8),
            Layer("Softmax", units=2)],
        n_iter=50,
        n_stable=10,
        batch_size=25,
        learning_rate=0.002,
        learning_rule="momentum",
        valid_size=0.1,
        verbose=1)

# Create the model with randomized search
rs_nn = RandomizedSearchCV(nn, param_distributions={
    'learning_rate': stats.uniform(0.001, 0.05),
    'hidden0__units': stats.randint(4, 12),
    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
rs_nn.fit(X, Y)
   
# summarize results
print("Best Parameters:\n", rs_nn.best_params_)
print("Best CV score:\n", rs_nn.best_score_)
print("Summary of best estimator", rs_nn.best_estimator_)

