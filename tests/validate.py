import numpy as np
from sigmapy.sigma_sets import SigmaSets
from scipy.stats import itemfreq

""""
Tests all the included sigma point methods to ensure that they correctly 
compute expected value integrals for all monomials up to a given order
problem.
"""

sets = SigmaSets()
x = np.zeros(5)
Px = np.eye(5)

exponents = np.array(np.meshgrid(*[range(5)]*5)).T.reshape(-1, 5).T

polynomials = {
    2 : exponents[:, exponents.sum(axis = 0) <= 2],
    3 : exponents[:, exponents.sum(axis = 0) <= 3],
    5 : exponents[:, exponents.sum(axis = 0) <= 5]
}

def test_set(name):
    X, wm, wc = sets.get_set(x, Px, set_name = name)

    order = sets.sigma_order[name]
    polys = polynomials[order]

    
    for i in range(polys.shape[1]):
        ks = polys[:,i]
        
        # Calculate expected value of monomial function E[x_0^k_1 ... x_4^k_4]
        # using a sigma point method
        sigma_int = np.dot(wm, np.prod(np.power(X, ks[:, np.newaxis]), axis = 0))
        
        # Compute the expectation integral analytically
        item_freq = itemfreq(ks)
        freq_dict = {}
        for k,v in zip(item_freq[:,0], item_freq[:,1]):
            freq_dict[k] = v 

        if freq_dict.keys() == set([0]):
            analytic_int = 1.
        elif (1 in freq_dict.keys()) or (3 in freq_dict.keys()) or (5 in freq_dict.keys()):
            analytic_int = 0.
        elif freq_dict.keys() == set([0,2]):
            analytic_int = 1.
        else :
            analytic_int = 3.

        print(ks)
        print(sigma_int, analytic_int)
        print()
        assert(np.abs(sigma_int - analytic_int) < 1e-14), "Error in sigma set: {}".format(name)
    
for name in sets.sigma_functions.keys():
    test_set(name)
    print("{} passed.".format(name))
        

