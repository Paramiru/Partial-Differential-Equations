import numpy as np

def keepBC3d(potential):
    # add boundary condition. set phi = 0 on the boundary
    potential[0], potential[-1] = 0, 0
    potential[1:-1, :, 0] = 0
    potential[1:-1, :, -1] = 0
    potential[1:-1, 0] = 0
    potential[1:-1,-1] = 0
    return potential

def keepBC2d(potential):
    potential[0], potential[-1] = 0, 0
    potential[:,0], potential[:,-1] = 0, 0
    return potential

def JacobiUpdate2d(potential, chargeDistribution, dx=1):
    new_potential = (np.roll(potential, 1, axis=0) + np.roll(potential, -1, axis=0) +\
        np.roll(potential, 1, axis=1) + np.roll(potential, -1, axis=1) +\
        dx**2*chargeDistribution) / 4
    return keepBC2d(new_potential)

def JacobiUpdate3d(potential, chargeDistribution, dx=1):
    new_potential = (np.roll(potential, 1, axis=0) + np.roll(potential, -1, axis=0) +\
        np.roll(potential, 1, axis=1) + np.roll(potential, -1, axis=1) +\
        np.roll(potential, 1, axis=2) + np.roll(potential, -1, axis=2) +\
        dx**2*chargeDistribution) / 6
    return keepBC3d(new_potential)

def GaussSeidelUpdate3d(l, potential, chargeDistribution, dx=1):
    for i in range(1, l-1):
        for j in range(1, l-1):
            for k in range(1, l-1):
                potential[i, j, k] = (
                    potential[i-1, j, k] + potential[i+1, j, k] +\
                    potential[i, j-1, k] + potential[i, j+1, k] +\
                    potential[i, j, k-1] + potential[i, j, k+1] +\
                    dx**2*chargeDistribution[i, j, k]
                    )/6
    return potential

def overRelaxation(w, l, potential, chargeDistribution, dx=1):
    for i in range(1, l-1):
        for j in range(1, l-1):
            for k in range(1, l-1):
                potential[i, j, k] = (1-w)*potential[i,j,k] +\
                    w*(potential[i-1, j, k] + potential[i+1, j, k] +\
                    potential[i, j-1, k] + potential[i, j+1, k] +\
                    potential[i, j, k-1] + potential[i, j, k+1] +\
                    dx**2*chargeDistribution[i, j, k]
                    )/6
    return potential
