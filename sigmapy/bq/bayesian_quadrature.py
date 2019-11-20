import GPy
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

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
        self.n = len(lengthscales)
        # Create a kernel
        self.kernel = GPy.kern.RBF(input_dim = self.n,
                                   variance = alpha**2,
                                   lengthscale = lengthscales,
                                   ARD = True)
        # Diagonal covariance matrix
        self.Lambda = np.diag(lengthscales**2)
        # Determinant of covariance matrix
        self.Lambda_det = np.prod(np.diag(self.Lambda))
        # Inverse covariance matrix
        self.Lambda_inv = 1. / self.Lambda
        # Determinant of inverse covariance matrix
        self.Lambda_inv_det = 1. / self.Lambda_det
        # Diagonal of  S = Lambda + I
        self.S = self.Lambda + np.eye(self.n)
        # Determinant of S 
        self.S_det = np.prod(np.diag(self.S))
        # Inverse of S
        self.S_inv = 1. / self.S
        # c is a constant such that c*k(x,x') is a Gaussian PDF
        self.c = 1. / (np.sqrt((2.*np.pi)**(self.n) * (self.Lambda_inv_det))*alpha**2)

        
   
    
    def get_weights(self, X, sigma = 0.):
        """
        Given a set of m quadrature points :math:`x_i \\in \\mathbb{R}^n`
        this function computes a corresponding vector of quadrature weights 
        :math:`\pmb{w}` via
        
        .. math:: 
           (K + \\sigma^2 I) \pmb{w} = \pmb{z}

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
        
        # Left hand side
        K = self.kernel.K(X,X)

        print(self.kernel.K(X,X))
        print()
        print(self.kernel.K(X))
        quit()
        print(K)
        A = K + (sigma**2 * np.eye(K.shape[0]))
        # Right hand side
        z = self.__int_gaussian_product__(X) * self.c
        # Solve for weights
        w = np.linalg.solve(A,z)
        return w
        



    def __int_gaussian_product__(self, X):
        """
        Given a set of m points :math:`x_i \\in \\mathbb{R}^n`, this 
        function computes integrals of products of of Gaussian PDF's. 

         .. math:: 
           \pmb{c} = \\begin{bmatrix}
           \\int_{\\mathbb{R}^n} 
           \\mathcal{N}(\pmb{x} | \pmb{x_1}, \\Lambda) 
           \\mathcal{N}(\\pmb{x} | \\pmb{0}, I) d \pmb{x} \\\\
           \\vdots \\\\
           \\int_{\\mathbb{R}^n} 
           \\mathcal{N}(\pmb{x} | \pmb{x_m}, \\Lambda)
           \\mathcal{N}(\pmb{x} | \pmb{0}, I) d \pmb{x}
           \\end{bmatrix}

        
        The i-th integral can be computed in closed form by

        .. math:: 
           c_i = (2 \\pi)^{-\\frac{n}{2}} 
           |\\Lambda_1 + I|^{-\\frac{1}{2}}
           \\exp \\left ( -\\frac{1}{2} \pmb{x_i}^T 
           (\\Lambda + I)^{-1} \pmb{x_i} \\right )

        Args:
           X (np.array(n, m): Array of points

        Returns:
           np.array(m): Array of integrals c

        """

        # Dimension
        n = float(self.n)
        # Determinant of S 
        S_det = self.S_det
        # Inverse of S
        S_inv = self.S_inv

        c0 = (2.*np.pi)**(-n/2.)
        c1 = S_det**(-1./2.)
        c2 = np.exp(-0.5 * (X @ S_inv @ X.T).sum(axis = 1))
        c = c0 * c1 * c2
        
        return c




