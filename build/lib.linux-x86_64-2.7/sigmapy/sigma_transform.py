import numpy as np
import matplotlib.pyplot as plt
from sigmapy.sigma_sets import SigmaSets
from sigmapy.normal import Normal

class SigmaTransform(object):
    
    """
    Estimates the mean and covariance of a transformed random variable
    using a sigma point method.

    Parameters
    ----------

    sqrt_method : function(np.array), default=np.linalg.cholesky
        Matrix square root function

    Attributes
    ----------

    points : :class:`.SigmaSets`
        Object for computing points and weights for various sigma point methods.
    
    """

    def __init__(self, sqrt_method):
        self.points = SigmaSets(sqrt_method)

        
    def do_transform(self, x, Px, F, mean_fn=None, add_x=None,
                     add_y=None, **sigma_args):
        """ 
        Given a nonlinear function :math:`f : \mathbb{R}^n \\to \mathbb{R}^m`,
        estimate the joint distribution of 
        :math:`[\pmb{x} \; \pmb{y}]` where

        .. math:: \pmb{y} = f(\pmb{x}) + \pmb{q}

        with :math:`\pmb{x} \\sim N(\pmb{x}, P_x)` and :math:`\pmb{q} 
        \\sim N(\pmb{0},Q)` using a sigma point method.

        Parameters
        ----------

        x : np.array(n)
           Prior mean vector

        Px : np.array(n,n) 
           Prior covariance matrix

        F : callable(np.array(n,N))
           Matrix form of the nonlinear function. It accepts an 
           :math:`n \\times N` array X with sigma points :math:`\pmb{\chi_i}`
           as columns. It returns an :math:`m \\times N` array Y with 
           transformed sigma points as columns:

           .. math:: Y = F(X) = [f(\pmb{\chi_0}), \; \cdots, \; f(\pmb{\chi_{N-1}})]           

        mean_fn : callable  (X, weights), optional
           Function that computes the mean of the provided sigma points
           and weights. Use this if your state variable contains nonlinear
           values such as angles which cannot be summed.

        add_x : callable (x1, x2), optional
           Function that computes the sum of state variables x1 and x2. 
           Useful for quantities like angles that can't simply be summed. 

        add_y : callable (y1, y2), optional
           Function that computes the transformed variables y1 and y2. 
           Useful for quantities like angles that can't simply be summed. 

        sigma_args : additional keyword arguments, optional
           Sigma point arguments such as the sigma set type and scaling 
           parameters. merwe sigma points are used by default.

        Returns
        -------

        y : np.array(m)
            Estimated mean of the transformed random variable y

        Py : np.array(m,m)
            Estimated covariance of the transformed random variable y

        Pxy : np.array(n,m)
            Estimated cross-covariance of the transformation

        X : np.array(n, N)
            Sigma point set used to estimate y, Py, and Pxy. The number
            of sigma points N depends on the method used.

        wm : np.array(N)
           Mean weights. N is the number of sigma points. 

        wc : np.array(N)
           Covariance weights. N is the number of sigma points.

        """

        if np.isscalar(x):
            x = np.asarray([x])
            n = 1
        else:
            # State dimension
            n = len(x)

        if  np.isscalar(Px):
            Px = np.eye(n)*Px
        else:
            Px = np.atleast_2d(Px)

        
        # Get sigma points
        X, wm, wc = self.points.get_set(x, Px, **sigma_args)
        
        # Columns are transformed sigma points
        Y = f(X)
        # Dimensions
        n, m = Y.shape

        # Compute transformed mean
        try:
            if mean_fn is None:
                y = np.dot(Y, wm)    
            else:
                y = y_mean(Y, wm)
        except:
            raise

        # Compute covariance
        if add_y is np.add or add_y is None:
            ry = Y - y[:, np.newaxis]
            Py = ry @ np.diag(wc) @ ry.T
        else:
            Py = np.zeros((n, n))
            for k in range(m): 
                ry = add_y(Y[k], -y)
                Py += wc[k] * np.outer(ry, ry)

        # Compute cross covariance
        if add_x is np.ad or add_x is None:
            rx = X - x[:, np.newaxis]
            Pxy = rx @ np.diag(wc) @ ry.T
        else:
            Pxy = np.zeros((m, n))
            for k in range(m):
                rx = add_x(X[k], x)
                Pxy += wc[k] * np.outer(rx, ry)

        joint_mean = np.block([x,y])
        joint_cov = np.block([[Px, Pxy],[Pxy.T, Py]])
        joint_dist = Normal(joint_mean, joint_cov)
        
        return joint_dist, X, wm, wc  

    
       
