import numpy as np

def int_pix(data, x, y):
    """
    Return the spectrum at a given pixel.

    Parameters:

    data - 3d array
    x - int
    y - int

    Returns:

    spec - 1d array
    """
    
    return data.T[x][y]

def zeroth(data, d_v):
    """
    Calculate zeroth moment for a data set.
    
    Parameters:
    
    data - 3d array
    d_v - float, units: m/s
    
    Returns:
    
    M_0 - 2d array
    """
    
    return np.sum(data, axis=0)*d_v

def first(data, rad_v):
    """
    Calculate first moment for a data set.
    
    Parameters:
    
    data - 3d array
    rad_v - 1d array, units: m/s
    
    Returns:

    M_1 - 2d array
    """
    
    return (np.dot(data.T, rad_v)/np.sum(data.T, axis=2)).T

def second(data, rad_v, M_1):
    """
    Calculate second moment for a data set.
    
    Parameters:
    
    data - 3d array
    rad_v - 1d array, units: m/s
    M_1 - 2d array
    
    Returns:
    
    M_2 - 2d array
    """
    
    M_2 = np.zeros([data.T.shape[0], data.T.shape[1]])
    for i in range(M_2.shape[0]):
        for j in range(M_2.shape[1]):
            spec = int_pix(data, i, j)
            M_2[i][j] = np.sqrt(np.dot((rad_v-M_1.T[i][j])**2, spec)/np.sum(spec))
    
    return M_2.T
