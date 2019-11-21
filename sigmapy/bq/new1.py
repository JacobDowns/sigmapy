import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
from scipy.stats import norm, multivariate_normal
from sigmapy.bq.bayesian_quadrature import BayesianQuadrature


### Integrate Gaussian process the standard way
######################################################################

np.random.seed(300)

# Training data
num_points = 60
X = np.random.uniform(-2.,2.,(num_points, 2))
Y = (np.sin(X[:,0])*np.cos(X[:,1]) + X[:,1]**2).reshape((num_points,1)) + 1.

# Squared exponential kernel
alpha = 1.
lengthscales = np.array([1.,1.])
kernel = GPy.kern.RBF(input_dim=2, variance = alpha**2, lengthscale=lengthscales, ARD = True)

# Gaussian process 
gp = GPy.models.GPRegression(X,Y,kernel, noise_var = 0.)

# Weighting function
m = np.array([0.,0.])
P = np.eye(2)

#w = multivariate_normal.pdf(X_grid, mean = m, cov = P).reshape(xx.shape)

# Weighted GP mean
def w_f(x, y):
    points = np.c_[x, y]
    f, v = gp.predict(points)
    f = f.flatten()
    w = multivariate_normal.pdf(points, mean = m, cov = P)
    return f*w

xx, yy = np.meshgrid(np.linspace(-4.,4., 100), np.linspace(-4.,4., 100))
z = w_f(xx.flatten(), yy.flatten())
z = z.reshape(xx.shape)
plt.imshow(z, extent = (-4., 4., -4., 4.))
plt.plot(X[:,0], X[:,1], 'ko')
plt.colorbar()
plt.show()

# Compute the integral via standard numerical quadrature
v1 = dblquad(w_f, -75., 75., lambda x : -75., lambda x: 75.,
             epsabs=1e-7,
             epsrel=1e-7)

print("int f(x)w(x)", v1)


### Integrate Gaussian process via a custom quadrature rule
######################################################################


bq = BayesianQuadrature(alpha**2, lengthscales, m, P)


#print(bq.__int_gaussian_product__(np.array([[0.]]), bq.Lambda + np.eye(1)))
#quit()

w, int_var = bq.get_weights(X)
print(int_var)
#print(w)
#print(Y)
print(np.dot(w, Y.flatten()))
