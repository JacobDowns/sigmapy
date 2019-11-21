import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import quadrature
from scipy.stats import norm, multivariate_normal
from sigmapy.bq.bayesian_quadrature import BayesianQuadrature


### Integrate Gaussian process the standard way
######################################################################

np.random.seed(300)

# Training data
X = np.random.uniform(-3.,3.,(12,1))
Y = np.sin(X) + X**2

# Squared exponential kernel
alpha = 1.
lengthscale = np.array([1.])
kernel = GPy.kern.RBF(input_dim=1, variance = alpha**2, lengthscale=lengthscale, ARD = True)

# Gaussian process 
gp = GPy.models.GPRegression(X,Y,kernel, noise_var = 0.)

# Mean and covariance of weight function
m = -0.3
P = 1.5
# Weighted GP mean
def weighted_f(x):
    f, v = gp.predict(x.reshape(len(x),1))
    f = f.flatten()
    w = multivariate_normal.pdf(x, mean = m, cov = P)
    return f*w

# Compute the integral via standard numerical quadrature
v1 = quadrature(weighted_f, -200, 200., maxiter = 2000, tol = 1e-12, rtol = 1e-12)

print("int f(x)w(x)", v1)

### Integrate Gaussian process via a custom quadrature rule
######################################################################


bq = BayesianQuadrature(alpha**2, lengthscale, m, P)


#print(bq.__int_gaussian_product__(np.array([[0.]]), bq.Lambda + np.eye(1)))
#quit()

w, int_var = bq.get_weights(X)
print(int_var)
#print(w)
#print(Y)
print(np.dot(w, Y.flatten()))
