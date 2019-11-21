import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import GPy
from scipy.stats import multivariate_normal

class BayesianQuadrature(object):
    
    """
    Computes weights for Bayesian quadrature rules for integrals of the 
    form 

    .. math:: 
       \\int_{\\mathbb{R}^n} g(\pmb{x}) N(\pmb{x} | \pmb{m}, P) d \pmb{x} 
       \\approx \sum_i w_i g(\pmb{x_i}) 

    where the points :math:`\pmb{x_i} \\in \\mathbb{R}^n` are training 
    points for a Gaussian process with mean function :math:`g(\pmb{x})` 
    and a squared exponential covariance function:

    .. math:: 
       k(\pmb{x}, \pmb{x}') = \\alpha^2 \\exp 
       \\left [ -\\frac{1}{2} (\pmb{x} - \pmb{x}')^T \\Lambda^{-1}  
       (\pmb{x} - \pmb{x}') \\right ]

    Here :math:`\\alpha^2` is kernel variance. :math:`\\Lambda` 
    is a diagonal matrix of squared length scales:

    .. math:: 
       \\Lambda = \\text{diag} ([\\ell_1^2, \\ell_2^2, \\cdots, \\ell_n^2])


    Args:
           alpha (int): Kernel variance

           lengthscales (np.array(n)): Array of length scales

    .. automethod:: __int_gaussian_product__

    """

    def __init__(self, alpha, lengthscales):

        # Dimension
        n = len(lengthscales)
        # Covariance kernel
        self.kernel = GPy.kern.RBF(input_dim = n,
                                   variance = alpha**2,
                                   lengthscale = lengthscales,
                                   ARD = True)

        # Covariance matrix
        self.Lambda = np.diag(lengthscales**2)
        # Determinant of inverse covariance matrix
        Lambda_inv_det = np.prod(1. / np.diag(self.Lambda))
        # c is a constant such that c*k(x,x') is a Gaussian PDF
        self.c = 1. / (np.sqrt((2.*np.pi)**(n) * (Lambda_inv_det))*alpha**2)
        
   
    
    def get_weights(self, X, sigma = 0.):
        """
        Given a set of m quadrature points :math:`x_i \\in \\mathbb{R}^n`
        this function computes a corresponding vector of quadrature weights 
        :math:`\pmb{w}` via
        
        .. math:: 
           (K + \\sigma^2 I) \pmb{w} = \pmb{k}

        with

        .. math:: 
           K = 
           \\begin{bmatrix}
           k(\pmb{x_1}, \pmb{x_1}) & \cdots & k(\pmb{x_1}, \pmb{x_m}) \\\\
           \\vdots & \\ddots & \\vdots \\\\
           k(\pmb{x_m}, \pmb{x_1}) & \cdots & k(\pmb{x_m}, \pmb{x_m})
           \\end{bmatrix}

        .. math:: 
           \pmb{z} = 
           \\begin{bmatrix}
           \\int_{\\mathbb{R}^n} k(\pmb{x}, \pmb{x_1}) \\mathcal{N}(\pmb{x} | \pmb{0}, I) \\; d \pmb{x} \\\\
           \\vdots \\\\
           \\int_{\\mathbb{R}^n} k(\pmb{x}, \pmb{x_m}) \\mathcal{N}(\pmb{x} | \pmb{0}, I) \\; d \pmb{x}  
           \\end{bmatrix}

        Args:
           X (np.array(n, m): Array of points

        Returns:
           np.array(m): Array of weights w

        """

        print(self.Lambda)
        
        # Left hand side
        K = self.kernel.K(X)
        A = K + (sigma**2 * np.eye(K.shape[0]))
        # Right hand side
        k = self.__int_gaussian_product__(X, self.Lambda) / self.c
        # Solve for weights
        w = np.linalg.solve(A, k)
        # Compute the variance of the integration rule
        v1 = self.__int_gaussian_product__(np.array([[0.]]),
                                           self.Lambda  +np.eye(self.Lambda.shape[0]))                                            
        v1 /= self.c
        v2 = k @ w
        
        print(v1, v2, v1 - v2)
        return w
        



    def __int_gaussian_product__(self, X, Lambda):
        """
        Given a set of m points :math:`x_i \\in \\mathbb{R}^n`, this 
        function computes integrals of products of of Gaussian PDF's. 

         .. math:: 
           \\begin{bmatrix}
           \\int_{\\mathbb{R}^n} 
           \\mathcal{N}(\pmb{x} | \pmb{x_1}, \\Lambda) 
           \\mathcal{N}(\\pmb{x} | \\pmb{0}, I) d \pmb{x} \\\\
           \\vdots \\\\
           \\int_{\\mathbb{R}^n} 
           \\mathcal{N}(\pmb{x} | \pmb{x_m}, \\Lambda)
           \\mathcal{N}(\pmb{x} | \pmb{0}, I) d \pmb{x}
           \\end{bmatrix} =
           \\begin{bmatrix}
           \\mathcal{N}(\pmb{x_1} | \pmb{0}, \\Lambda + I) \\\\
           \\mathcal{N}(\pmb{x_2} | \pmb{0}, \\Lambda + I) \\\\
           \\vdots \\\\
           \\mathcal{N}(\pmb{x_i} | \pmb{0}, \\Lambda + I) 
           \\end{bmatrix}

        Args:
           X (np.array(m, n): Array of m points in :math:`\\mathbb{R}^n`

        Returns:
           np.array(m): Integrals for each point

        """

        print("Lamb", Lambda)
        n = X.shape[1]
        norm = multivariate_normal(np.zeros(n), Lambda + np.eye(n))
        return norm.pdf(X)



