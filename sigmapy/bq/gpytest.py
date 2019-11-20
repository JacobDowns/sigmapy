import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import quadrature
from scipy.stats import norm, multivariate_normal
from sigmapy.bq.bayesian_quadrature import BayesianQuadrature


### Integrate Gaussian process the standard way
######################################################################

np.random.seed(100)

# Training data
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + X**2

# Squared exponential kernel
alpha = 1.
lengthscale = np.array([1.])
kernel = GPy.kern.RBF(input_dim=1, variance = alpha**2, lengthscale=lengthscale, ARD = True)

# Gaussian process 
m = GPy.models.GPRegression(X,Y,kernel, noise_var = 0.)

# Weighted GP mean
def weighted_g(x):
    g, v = m.predict(x.reshape(len(x),1))
    g = g.flatten()
    w = multivariate_normal.pdf(x)
    return g*w

# Plot the weighted function w(x)*g(x)
xs = np.linspace(-5.,5.,500)
plt.plot(xs, weighted_g(xs))
plt.show()

# Compute the integral via standard numerical quadrature
v1 = quadrature(weighted_g, -35, 35., maxiter = 500)
print(v1)


### Integrate Gaussian process via a custom quadrature rule
######################################################################

bq = BayesianQuadrature(alpha**2, lengthscale)
w = bq.get_weights(X)
print(np.dot(w, Y))
print(w)
quit()

def p(x, mean):
    p1 = multivariate_normal.pdf(x)
    p2 = multivariate_normal.pdf(x, mean = mean)
    return p1*p2

v = quadrature(lambda x : p(x, mean = X[1,0]), -50., 50., maxiter = 500)
print(v)
