import numpy as np
from matplotlib import pyplot as plt
from algorithms import GaussSeidelUpdate3d, JacobiUpdate3d, keepBC3d, keepBC2d, overRelaxation

class Poisson:
    def __init__(self, l:50, accuracy=float(1e-3)):
        self.l = l
        self.rng = np.random.default_rng(seed=14122000)
        self.accuracy = accuracy
        self.chargeDistribution = np.zeros(shape=(self.l, self.l))

    def initialisePotential3d(self):
        grid = self.rng.random(size=(self.l, self.l, self.l))
        return keepBC3d(grid)

    def initialisePotential2d(self):
        grid = self.rng.random(size=(self.l, self.l))
        return keepBC2d(grid)

    def plot3dPotential(self):
        potential = self.potential[self.l//2]
        plt.imshow(potential)
        plt.show()

    def solvePoisson(self, update, w=1):
        error, epoch = 1, 1
        while error > self.accuracy:
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}")
                print(f"Error: {error}")
            previousPotential = self.potential.copy()
            if update == GaussSeidelUpdate3d:
                update(self.l, self.potential, self.chargeDistribution)
            elif update == overRelaxation:
                self.potential = update(w, self.l, self.potential, self.chargeDistribution)
            else:
                self.potential = update(self.potential, self.chargeDistribution)
            error = np.abs(np.sum(self.potential - previousPotential))
            epoch+=1
        return epoch

    def storeData(self, fieldx, fieldy, fieldz):
        f=open('spins.csv','w')
        f.write(f'i,j,k,Ex,Ey,Ez\n')
        for i in range(self.l):
            for j in range(self.l):
                for k in range(self.l):
                    f.write('%d,%d,%d,%lf,%lf,%lf\n'%(i,j,k,fieldx[i,j,k], fieldy[i,j,k], fieldz[i,j,k]))
        f.close()
    