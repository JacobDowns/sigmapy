import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix

class WeightComputer(object):
    
    """
    Computes weights for Bayesian quadrature rules where the integrand
    is a Gaussian process with mean function :math:`m(\pmb{x})` and squared
    exponential covariance function 

    .. math:: 
       k(\\pmb{x}, \\pmb{x}') = \\alpha^2 \\exp 
       \\left [ -\\frac{1}{2} (\\pmb{x} - \\pmb{x}')^T \\Lambda^{-1}  (\\pmb{x} - \\pmb{x}') \\right ]

    where :math:`\\alpha^2` is the process variance and :math:`\\Lambda` 
    is the covariance matrix. 

    Args:
           alpha (int): Process standard deviation

           lambda_diag (np.array(n)): Diagonal entries of covariance 
              matrix 

    .. automethod:: __int_gaussian_product__

    """

    def __init__(self, alpha, lambda_diag):
        
        # Process variance
        self.alpha = alpha
        # Dimension
        self.n = len(lambda_diag)
        # Covariance matrix
        self.Lambda = np.diag(lambda_diag)
        # Determinant of covariance matrix
        self.Lambda_det = np.prod(lambda_diag)
        # Inverse covariance matrix
        self.Lambda_inv = np.diag(1. / lambda_diag)
        # Determinant of inverse covariance matrix
        self.Lambda_inv_det = 1. / self.Lambda_det
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

        ### Assemble equation Aw = z
        ##############################################################

        # Left hand side
        K = self.kernel(X)
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
           \\mathcal{N}(\pmb{x} | \pmb{x_m}, \\Lambda_1)
           \\mathcal{N}(\pmb{x} | \pmb{0}, I) d \pmb{x}
           \\end{bmatrix}

        
        The i-th integral can be computed in closed form by

        .. math:: 
           c_i = (2 \\pi)^{-\\frac{n}{2}} 
           |\\Lambda_1 + \\Lambda_2|^{-\\frac{1}{2}}
           \\exp \\left ( -\\frac{1}{2} \pmb{x_i}^T 
           (\\Lambda_1 + \\Lambda_2)^{-1} \pmb{x_i} \\right )

        Args:
           X (np.array(n, m): Array of points

        Returns:
           np.array(m): Array of integrals c

        """

        # Dimension
        n = float(self.n)
        # Diagonal of  S = Lambda + I
        S_diag = np.diag(self.Lambda) + 1.
        # Determinant of S 
        S_det = np.prod(S_diag)
        # Inverse of S
        S_inv = np.diag(1. / S_diag)

        c0 = (2.*np.pi)**(-n/2.)
        c1 = S_det**(-1./2.)
        c2 = np.exp(-0.5 * (X.T @ S_inv @ X).sum(axis = 1))
        c = c0 * c1 * c2
        
        return c
    

    def kernel(self, X):
        """
        Given a set of m points :math:`\pmb{x_i} \\in \\mathbb{R}^n`
        compute the matrix 

        .. math:: 
           K = 
           \\begin{bmatrix}
           k(\pmb{x_1}, \pmb{x_1}) & \cdots & k(\pmb{x_1}, \pmb{x_m}) \\\\
           \\vdots & \\ddots & \\vdots \\\\
           k(\pmb{x_m}, \pmb{x_1}) & \cdots & k(\pmb{x_m}, \pmb{x_m})
           \\end{bmatrix}

        Args:
           X (np.array(n, m): Array of points

        Returns:
           np.array(m, m): K

        """
        D = squareform(pdist(X.T, metric =
                             lambda x, y : np.sqrt((x-y).T @ self.Lambda_inv @ (x-y))))

        return self.alpha**2 * np.exp(-0.5*D)

import matplotlib.pyplot as plt
bq = WeightComputer(alpha = 1., lambda_diag = np.ones(2))
X = np.array([[1.,-1], [2., -2.], [0., 0.]]).T
print(bq.get_weights(X))

#print(bq.__int_k_w__(X))



