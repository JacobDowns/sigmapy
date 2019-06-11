import numpy as np
from numpy.linalg import cholesky, inv

class SigmaSets(object):
    
    """
    Generates sigma points and weights using one of several available 
    methods.

    Parameters
    ----------

    sqrt_method : function(ndarray)
        The matrix square root used to compute sigma points. 

    add : callable (x, y), optional
        Function that computes the sum of x and y.

    Attributes
    ----------

    sigma_functions : dict
        Dictionary mapping sigma point set names to functions for computing
        those sets. 

    sigma_order : dict
        Dictionary mapping sigma point set names to their order of accuracy

    """

    def __init__(self, sqrt_method = None, add = None):

        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if add is None:
            self.add = np.add
        else:
            self.add = add
            
        # Available sigma point sets
        self.sigma_functions = {}
        self.sigma_functions['merwe'] = self.get_set_merwe
        self.sigma_functions['menegaz'] = self.get_set_menegaz
        self.sigma_functions['li'] = self.get_set_li
        self.sigma_functions['mysovskikh'] = self.get_set_mysovskikh
        self.sigma_functions['julier'] = self.get_set_julier
        self.sigma_functions['simplex'] = self.get_set_simplex
        self.sigma_functions['hermite'] = self.get_set_hermite

        # Method order
        self.sigma_order = {}
        self.sigma_order['merwe'] = 3
        self.sigma_order['menegaz'] = 2
        self.sigma_order['li'] = 5
        self.sigma_order['mysovskikh'] = 3
        self.sigma_order['julier'] = 3
        self.sigma_order['simplex'] = 2
        self.sigma_order['hermite'] = 3
        
    
    def get_set(self, x, Px, **sigma_args):

        """
        Computes the sigma point and weight sets using one of several
        available methods. 

        Parameters
        ----------

        x : scalar, or np.array
           Mean vector of length n.

        Px : scalar, or np.array
           Covariance matrix. If scalar, is treated as eye(n)*P.

        set_name : string, default='mwer'
           The name of the sigma point set to compute. 

        **sigma_args : scalar scaling variables
           Additional parameters used to scale the sigma points. These 
           vary from method to method.

        Returns
        -------

        X : np.array, of size (n, N)
            Two dimensional array of sigma points. Each column is a 
            single sigma point. 

        wm : np.array
            Mean weights

        wc : np.array
            Covariance weights

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

        # Sigma point set
        set_name = 'merwe'
        if 'set_name' in sigma_args:
            set_name = sigma_args['set_name']
            sigma_args.pop('set_name')

        # Get sigma points for N(0, I)
        X, wm, wc = self.sigma_functions[set_name](n, **sigma_args)
        # Change variables to get sigma points for N(x, Px)
        X = self.add(x[:,None].repeat(X.shape[1], axis = 1), self.sqrt(Px)@X)

        return X, wm, wc


    def get_set_merwe(self, n, **scale_args):
        """
        Generates sigma points and weights according to the third order 
        method in [1]_.

        Parameters
        ----------

        n : int
            Dimensionality of the state. 2n+1 points will be generated.


        alpha : float, default = 0.5
            Scaling parameter


        beta : float, default = 2.
            Scaling paramter

        kappa : float, default=0.
            Scaling parameter

        Returns
        -------

        X : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            Mean weights. 

        wc : np.array
            Covariance Weights.

        References
        ----------
        .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)

        """

        alpha = 0.5
        if 'alpha' in scale_args:
            alpha = scale_args['alpha']
        beta = 2.
        if 'beta' in scale_args:
            beta = scale_args['beta']
        kappa = 3. - n
        if 'kappa' in scale_args:
            kappa = scale_args['kappa']

        lambda_ = alpha**2*(n + kappa) - n
        

        ### Sigma points
        X = np.sqrt(n + lambda_)*np.block([np.zeros(n)[:,None], np.eye(n), -np.eye(n)])

        
        ### Weights
        c = 1. / (2.*(n + lambda_))
        wc = np.full(2*n + 1, c)
        wm = np.full(2*n + 1, c)
        wm[0] =  lambda_ / (n + lambda_)
        wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
        
        return X, wm, wc


    def get_set_julier(self, n, **scale_args):
        """
        Generates sigma points and weights according to the method in 
        third order method in [2]_. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. 2n+1 points will be generated.

        kappa : float, default=0.
            Scaling factor

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            Mean weights

        wc : np.array
            Covariance weights

        References
        ----------
        .. [2] S. Julier and J. Uhlmann "New extension of the Kalman filter
           to nonlinear systems" 

       """

        
        kappa = 3. - n
        if 'kappa' in scale_args:
            kappa = scale_args['kappa']
        

        # Sigma points
        X = np.sqrt(n + kappa)*np.block([np.zeros(n)[:,None], np.eye(n), -np.eye(n)])
        
        # Weights
        c = 1. / (2.*(n + kappa))
        wm = np.full(2*n + 1, c)
        wm[0] = kappa / (n + kappa)
        
        return X, wm, wm

    
    def get_set_menegaz(self, n, **scale_args):
        """ 
        Computes the sigma points and weights using the second order 
        method in [3]_.

        Parameters
        ----------

        n : int
            Dimensionality of the state. n+1 points will be generated.

        w0 : scalar
           A scaling parameter with 0 < w0 < 1

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            Mean weights

        wc : np.array
            Covariance weights


        References
        ----------
        .. [3] H.M. Menegaz et al. "A new smallest sigma set for the Unscented 
           Transform and its applications on SLAM" 

        """

        w0 = 0.5
        # If the first weight is defined
        if 'w0' in scale_args:
            w0 = scale_args['w0']
            if w0 >= 1.0 or w0 <= 0.0:
                raise ValueError("w0 must be between 0 and 1")


        ### Sigma point set
        alpha = np.sqrt((1. - w0) / n)
        C = self.sqrt(np.diag(np.ones(n), 0) - (alpha**2)*np.ones((n, n)))
        C_inv = inv(C)

        W = np.diag(np.diag(w0*(alpha**2)*C_inv @ np.ones((n,n)) @ C_inv.T), 0)
        W_sqrt = self.sqrt(W)

        X = np.zeros((n, n+1))
        X[:,0] =  -(alpha / np.sqrt(w0))*np.ones(n)
        X[:,1:] = C @ inv(W_sqrt)
        X = X.T

        
        ### Weights
        w = np.zeros(n+1)
        w[0] = w0
        w[1:] = np.diag(W, 0)

        return X.T, w, w


    def get_set_simplex(self, n, **scale_args):
        """
        Generates sigma points and weights according to the second order 
        simplex method presented in [4]_.

        Parameters
        ----------

        n : int
            Dimensionality of the state. n+1 points will be generated.

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            Mean weights

        wc : np.array
            Covariance weights


        References
        ----------
        .. [4] Phillippe Moireau and Dominique Chapelle "Reduced-Order
           Unscented Kalman Filtering with Application to Parameter
           Identification in Large-Dimensional Systems"

        """

        # Generate sigma points
        lambda_ = n / (n + 1)
        Istar = np.array([[-1/np.sqrt(2*lambda_), 1/np.sqrt(2*lambda_)]])
        for d in range(2, n+1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1. / np.sqrt(lambda_*d*(d + 1))
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row]

        X = np.sqrt(n)*Istar

        # Generate weights
        wm = np.full(n + 1, 1. / (n+1.))
        
        return X, wm, wm


    def get_set_li(self, n, **scale_args):
        r""" 
        Computes the sigma points and weights for a modified version of 
        the fifth order method in [5]_. Setting the scaling parameter 
        :math:`r = \sqrt{3}` recovers the original method. This method 
        also requires :math:`n - r^2 -1 \neq 0`.

        Parameters
        ----------

         n : int
            Dimensionality of the state. 2n^2 + 1 points will be generated.

        r : scalar
           A scaling parameter with n - r^2 - 1 != 0

        Returns
        -------

        X : np.array, of size (n, 2n^2 + 1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance


        References
        ----------
        .. [5] Li, Z. et al. "A Novel Fifth-Degree Cubature Kalman Filter 
           for Real-Time Orbit Determination by Radar" 
        """

        r = np.sqrt(3./2.)
        # If the first weight is defined
        if 'r' in scale_args:
            r = slace_args['r']
            if n < 5 or abs(n - r**2 - 1.) < 1e-16:
                raise ValueError("This method requires n>=4 and n - r^2 - 1 != 0")
        

        # Weights

        # Coordinate for the first symmetric set
        r1 = (r*np.sqrt(n-4.))/np.sqrt(n - r**2 - 1.)
        # First symmetric set weight
        w2 = (4. - n) / (2. * r1**4)
        # Second symmetric set weight
        w3 = 1. / (4. * r**4)
        # Center point weight
        w1 = 1. - 2.*n*w2 - 2.*n*(n-1)*w3
        # Vector of weights
        w = np.block([w1, np.repeat(w2, 2*n), np.repeat(w3, 2*n*(n-1))])


        # Points
        
        # First fully symmetric set
        X0 = r1*np.eye(n)
        X0_s = np.block([X0, -X0])
        
        # Second fully symmetric set
        X1 = r*np.eye(n)
        indexes_i = []
        indexes_j = []
        for i in range(1,n):
            indexes_i.append(np.repeat([i],i))
            indexes_j.append(np.arange(0,i))
        indexes_i = np.concatenate(indexes_i).ravel()
        indexes_j = np.concatenate(indexes_j).ravel()
        P1 = X1[indexes_i, :].T + X1[indexes_j, :].T
        P2 = X1[indexes_i, :].T - X1[indexes_j, :].T
        X1_s = np.block([P1, P2, -P1, -P2])


        # Full set of points (columns are points)
        X = np.block([np.zeros(n)[:,None], X0_s, X1_s])

        return X, w, w
    

    def get_set_mysovskikh(self, n, **scale_args):
        """
        Computes the sigma points and weights for a fifth order cubature 
        rule due Mysovskikh, and outlined in [6]_. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. n^2 + 3n + 3 points will be generated.

        Returns
        -------

        X : np.array, of size (n, n^2 + 3n + 3)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            Mean weights

        wc : np.array
            Covariance weights

        References
        ----------
        .. [6] J. Lu and D.L. Darmofal "Higher-dimensional integration 
           with gaussian weight for applications in probabilistic design" 

        """

        # First set of points
        I = (np.arange(n)[:,None] + 1).repeat(n + 1, axis = 1).T
        R = (np.arange(n + 1) + 1)[:,None].repeat(n, axis = 1)
        A = -np.sqrt((n+1.) / (n*(n-I+2.)*(n-I+1.)))
        indexes = (I == R)
        A[indexes] = np.sqrt( ((n+1.)*(n-R[indexes]+1.)) / (n*(n-R[indexes]+2.)
))
        indexes = I > R
        A[indexes] = 0.
        

        # Second set of points
        ls = np.arange(n+1)[:,None].repeat(n+1)
        ks = (np.arange(n+1)[:,None].repeat(n+1, axis = 1).T).flatten() 
        indexes = ks < ls
        B = np.sqrt(n / (2.*(n-1.)))*(A[ks[indexes]] + A[ls[indexes]])

        # Full set
        #X = np.sqrt(n + 2.)*np.block([[np.zeros(n)], [A], [-A], [B], [-B]])
        X = np.block([[np.zeros(n)], [A], [-A], [B], [-B]])
        
        # Weights
        w0 = 2./(n+2.)
        w1 = (n**2 * (7. - n)) / (2.*(n + 1.)**2 * (n+2.)**2)
        w2 = (2.*(n-1.)**2) / ((n+1.)**2 * (n+2.)**2)
        w = np.block([w0, np.repeat(w1, 2*len(A)), np.repeat(w2, 2*len(B))])

        print(A.shape)
        print(B.shape)
        quit()
        
        return X.T, w, w 



    def get_set_hermite(self, n, **scale_args):
        """
        Computes the sigma points and weights for the third order Gauss
        Hermite method [7]_. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. 3^n points will be generated.

        Returns
        -------

        X : np.array, of size (n, 3^n)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance

        References
        ----------
        .. [7] Peng, Lijun et al.  "A New Sparse Gauss-Hermite Cubature 
           Rule Based on Relative-Weight-Ratios for Bearing-Ranging Target
           Tracking" 

        """

        # Sigma points
        X = np.array(np.meshgrid(*[[0., 1. , -1.]]*n)).T.reshape(-1, n).T
        # Mean and covariance weights
        js = (X**2).sum(axis = 0)
        wm = (2./3.)**(n-js) * (1./6.)**(js)
        wc = (2./3.)**(n-js) * (1./6.)**(js)

        X *= np.sqrt(3.)

        return X, wm, wm
