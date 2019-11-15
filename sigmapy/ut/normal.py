import numpy as np

class Normal(object):

    """
    Represents a multivariate normal distribution 

    .. math:: \pmb{u} \\sim N(\pmb{u_0}, P_u)

    with mean :math:`\pmb{u_0}` and covariance matrix :math:`P_u`.

    Parameters
    ----------

    u0 : np.array(n)
       Mean vector

    Pu : np.array(n,n) 
       Covariance matrix  

    """

    def __init__(self, u0, Pu):

        self.u0 = self.array_like(u0)
        self.Pu = self.array_like(Pu)
        self.u0[:] = self.u0
        self.Pu[:] = self.Pu


    def get_conditional(self, yo, Q):
        r""" 
        Computes a conditional distribution given a measurement
        :math:`\pmb{y_o} \in \mathbb{R}^m` with measurement noise 
        :math:`\pmb{q} \sim N(\pmb{0},Q)`. The mean and covariance of the 
        current distribution are partitioned based on the size of 
        :math:`\pmb{y_o}`:

        .. math:: 
           \pmb{u_0} = [\pmb{x} \; \pmb{y}]

        .. math:: 
           P_u = \begin{bmatrix}
                    P_x & P_y \\
                    P_y^T & P_{xy}
                 \end{bmatrix}
         
        Here :math:`\pmb{y} \in \mathbb{R}^m` and :math:`P_y \in \mathbb{R}^{m \times m}`. 
        Letting :math:`P_y' = P_y + Q`, the conditional distribution, 
        accounting for measurement noise, is given by

        .. math:: 
           \pmb{x} | \pmb{y_o} \sim N \left ( x + K[y_o - y],  P_x - K P_y' K^T \right )

        where :math:`K = P_{xy} P_y'`.

        Parameters
        ----------

        yo : np.array(n)
           Measurement or observation mean

        Q : np.array(n,n) 
           Prior covariance matrix

        Returns
        ----------
        
        cond_dist : :class:`.Normal`
           The normal, conditional distribution 

        """

        if np.isscalar(yo):
            yo = [yo]
            Q = np.atleast_2d(Q)
            m = 1
        else :
            m = len(yo)
            
        n = len(self.u0) - m

        # Partition the mean and covariance
        x = self.u0[0:n]
        mu = self.u0[n:]
        Px = self.Pu[0:n, 0:n]
        S = self.Pu[n:, n:] + Q
        C = self.Pu[n:, 0:m] 

        # Compute the conditional distribution
        K = C@np.linalg.inv(S)
        x_new = x + K@(yo - mu)
        Px_new = Px - K@S@K.T

        return Normal(x_new, Px_new)


    def get_marginal(self, inds):
        r""" 
        Computes a marginal distribution given indices of a subset of 
        variables. This just takes rows and columns of the variables
        corresponding to the indices. 

        Parameters
        ----------

        inds : np.array(k)
           Indices of variables to include in the marginal distribution


        Returns
        ----------
        
        marg_dist : :class:`.Normal`
           The resulting normal, marginal distribution

        """
        
        return Normal(self.u0[inds], self.Pu[inds][:inds])


    

       



        
                


        
        
