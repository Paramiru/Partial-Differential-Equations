import numpy as np
from algorithms import GaussSeidelUpdate3d, overRelaxation
from Poisson import Poisson
from matplotlib import pyplot as plt
import pandas as pd

class ElectricPoisson(Poisson):
    def __init__(self, l, ):
        super().__init__(l)
        self.potential = self.initialisePotential3d()
        self.chargeDistribution = self.getPointChargeGrid()

    def getPointChargeGrid(self):
        grid = np.zeros((self.l, self.l, self.l))
        grid[self.l//2,self.l//2,self.l//2] = 1
        return grid

    def getEfield(self, potential, dx=1):
        Ez = (np.roll(potential, 1, axis=0) - np.roll(potential, -1, axis=0))/(2*dx**2)
        Ey = (np.roll(potential, 1, axis=1) - np.roll(potential, -1, axis=1))/(2*dx**2)
        Ex = (np.roll(potential, 1, axis=2) - np.roll(potential, -1, axis=2))/(2*dx**2)
        return Ex, Ey, Ez

    def plotEfield(self):
        Ex, Ey, Ez = self.getEfield(self.potential)
        self.storeData(Ex, Ey, Ez)

        plt.imshow(Ex[self.l//2])
        plt.show()

        sliceEx = Ex[self.l//2]
        sliceEy = Ey[self.l//2]
        sliceEz = Ez[self.l//2]
        norm = np.sqrt(sliceEx**2 + sliceEy**2 + sliceEz**2)
        plt.quiver(sliceEx/norm, sliceEy/norm)
        plt.show()

        # sliceEx = sliceEx[15:35, 15:35]
        # sliceEy = sliceEy[15:35, 15:35]
        # sliceEz = sliceEz[15:35, 15:35]
        # norm = np.sqrt(sliceEx**2 + sliceEy**2 + sliceEz**2)
        # print(norm)
        # plt.quiver(sliceEx/norm, sliceEy/norm)
        # plt.show()

    def findMinimumW(self):
        ws = np.arange(1, 2, step=0.02)
        print(ws)
        steps = []
        for w in ws:
            print(f"Using w: {w}")
            self.potential = self.initialisePotential3d()
            steps.append(self.solvePoisson(update=overRelaxation, w=w))
        data = {'w': ws, 'steps':steps}
        pd.DataFrame.from_dict(data).to_csv('minimum_W_0.01.csv', index=False)
