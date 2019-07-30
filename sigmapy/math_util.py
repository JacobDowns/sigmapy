import numpy as np

def chebyshev_laguerre_cofs(nx, nc):
    
    """
    Computes coefficients for Chebyshev-Laguerre polynomials given by
    .. math:: L_{n_c}^{(a)} = (-1)^{n_c} t^{-a} e^t \frac{d^{n_c}}{dt^{n_c}}
    \left (t^{a + n_c} e^{-t} \right )
    with :math:`a = \frac{n_x}{2} - 1`.

    ----------
    nx: int 
        State variable dimension
    nc : int
        Degree of the Chebyshev-Laguerre polynomial
  
    Returns
    -------
    cofs : np.array(nc+1)
        Coefficients for the Laguerre Chebyshev-Laguerre polynomial of 
        order nc
    """
      
    # Differentiates an exponential polynomaial (non-integer powers)
    def dif(cofs, exps):
        cofs_new = np.zeros_like(cofs)
        cofs_new[1:] = cofs[:-1]*exps[:-1]
        return cofs_new

    a = (nx / 2.) - 1.
    exps = a + nc + - np.arange(nc+1)
    cofs = np.zeros_like(exps)
    cofs[0] = 1.
    print(a, nc)

    for i in range(nc):
        cofs = (-cofs + dif(cofs, exps))
        print(cofs)
        
    return (-1)**(nc)*cofs

def simplex_vertices(n):
    
    """
    Compute vertices of a regular simplex in n dimensions.

    ----------
    n: int 
        Dimension
  
    Returns
    -------
    A : np.array(n, n+1)
        Each column is a vertex on the regular simplex.
    """
    
    jj, mm = np.meshgrid(np.arange(1,n+2), np.arange(1,n+1))
    # Each column is a simplex vertex
    A = -np.sqrt((n+1.) / (n*(n-mm+2.)*(n-mm+1.)))
    indexes = (mm == jj)
    A[indexes] = np.sqrt(((n+1.)*(n-jj[indexes]+1.)) /
                          (n*(n-jj[indexes]+2.)))
    indexes = mm > jj
    A[indexes] = 0.

    return A

    

    
    
