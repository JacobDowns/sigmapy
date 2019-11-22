import numpy as np
import matplotlib.pyplot as plt
from sigmapy.bq.bayesian_quadrature import BayesianQuadrature

X = np.array([[0.,1.], [1.,0.], [0., -1.], [-1.,0.]])
plt.plot(X[:,0], X[:,1], 'ko')
plt.show()

alpha = 1.
lengthscales = np.array([1.,1.])
m = np.zeros(2)
P = np.eye(2)
bq = BayesianQuadrature(alpha**2, lengthscales, m, P)

def f(X):
    w, v = bq.get_weights(x)
    return v

