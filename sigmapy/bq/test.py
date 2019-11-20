import numpy as np
import gpflow as gpflow
from scipy.stats import multivariate_normal
from scipy.integrate import quadrature

np.random.seed(100)

xs = np.random.rand(10)[:,np.newaxis] - 0.5


ys = 2.0 * np.random.randn(10)[:,np.newaxis]

k = gpflow.kernels.SquaredExponential(1)

k1 = gpflow.kernels.SquaredExponential(2, np.array([2.,1.]), np.array([2.,1.]))

quit()
m = gpflow.models.GPR(xs, ys, k)
m.likelihood.variance = 1e-6
m.compile()
#opt = gpflow.train.ScipyOptimizer()
#opt.minimize(m)


#print(m.as_pandas_table())

import matplotlib.pyplot as plt




def w_g_vec(x):
    print("x", x)
    g, v = m.predict_f(x.reshape(len(x),1))
    g = g.flatten()
    w = np.array(list(map(lambda x_i : multivariate_normal.pdf(x_i), x)))
    #print("w", w)
    #print("g", g)
    return g*w

def w_g(x):
    print("x", x)
    g, v = m.predict_f([x])
    g = g[0,0]
    w = multivariate_normal.pdf(x)
    return w*g


#print(w_g(1.))
print(quadrature(w_g_vec, -20., 20., maxiter = 500))
xx = np.linspace(-20., 20., 50000)
yy =  w_g_vec(xx)
print(yy.sum() * (40. / 50e3))

quit()
#quit()

#plt.plot(xs, ys, 'ko')
plt.plot(xx, np.array(list(map(lambda x : w_g(x), xx))), 'b', lw=2)
#plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
#plt.xlim(-0.1, 1.1)
plt.show()

