from algorithms import GaussSeidelUpdate3d, keepBC2d, overRelaxation
from Poisson import Poisson
import numpy as np
from matplotlib import pyplot as plt

class MagneticPoisson(Poisson):
    def __init__(self, l, is2d=True):
        super().__init__(l)
        if is2d:
            self.potential = self.initialisePotential2d()
            self.chargeDistribution = self.getCurrentDistribution2d()
        else:
            self.potential = self.initialisePotential3d()
            self.chargeDistribution = self.getCurrentDistribution3d()

    def getCurrentDistribution2d(self):
        grid = np.zeros((self.l, self.l))
        grid[self.l//2, self.l//2] = 1
        return grid

    def getCurrentDistribution3d(self):
        grid = np.zeros((self.l, self.l, self.l))
        grid[:, self.l//2, self.l//2] = 1
        return grid

    def getBfield2d(self, potential, dx=1):
        Bx = -(np.roll(potential, 1, axis=0) - np.roll(potential, -1, axis=0))/(2*dx**2)
        By = (np.roll(potential, 1, axis=1) - np.roll(potential, -1, axis=1))/(2*dx**2)
        Bz = np.zeros(shape=(self.l, self.l))
        return Bx, By, Bz

    def plotBfield2d(self):
        Bx, By, _ = self.getBfield2d(self.potential)
        # divergence of B should be 0
        div = np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1) +\
            np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)
        print(f"Isclose: {np.isclose(np.max(np.abs(div)), 0)}")
        # print(f"\n\n\nDivergence: {div}")
        keepBC2d(Bx)
        # Bz is zero due to symmetry
        norm = np.sqrt(Bx**2 + By**2)
        plt.quiver(Bx/norm, By/norm)
        plt.show()
        Bx, By = Bx[15:35], By[15:35]
        norm = np.sqrt(Bx**2 + By**2)
        plt.quiver(Bx/norm, By/norm)
        plt.show()

        #TODO: FIX THIS SHIT
        # x = np.arange(1, 50)
        # y = np.arange(1, 50)
        
        # # Creating grids
        # X, Y = np.meshgrid(x, y)

        # plt.streamplot(X, Y, sliceBx, sliceBy)

    def getBfield3d(self, potential, dx=1):
        Bx = -(np.roll(potential, 1, axis=2) - np.roll(potential, -1, axis=2))/(2*dx**2)
        By = (np.roll(potential, 1, axis=1) - np.roll(potential, -1, axis=1))/(2*dx**2)
        Bz = np.zeros(shape=(self.l, self.l, self.l))
        return Bx, By, Bz

    def plotBfield3d(self):
        Bx, By, _ = self.getBfield3d(self.potential)
        # divergence of B should be 0
        div = np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1) +\
            np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)
        print(f"Isclose: {np.isclose(np.max(np.abs(div)), 0)}")
        # print(f"\n\n\nDivergence: {div}")
        sliceBx = Bx[self.l//2]
        sliceBy = By[self.l//2]
        keepBC2d(Bx)
        # Bz is zero due to symmetry
        norm = np.sqrt(sliceBx**2 + sliceBy**2)
        plt.quiver(sliceBy/norm, sliceBx/norm)
        plt.show()
        sliceBx, sliceBy = sliceBx[15:35], sliceBy[15:35]
        norm = np.sqrt(sliceBx**2 + sliceBy**2)
        plt.quiver(sliceBy/norm, sliceBx/norm)
        plt.show()
