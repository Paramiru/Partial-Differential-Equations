import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# noise between -0.1 and 0.1
# initialisation

class CahnHilliardSolver:
    def __init__(self, l=100, initialPhi=0, nstep=int(1e5)):
        self.l = l
        self.rng = np.random.default_rng()
        self.initialPhi = initialPhi
        self.phi = self.initialise_phi(initialPhi)
        self.nstep = nstep
        self.mu = np.zeros((self.l, self.l))
        self.compute_mu()

    def initialise_phi(self, initialPhi):
        # phi = self.rng.random() - 0.5
        noise = (self.rng.random(size=(self.l, self.l)) - 0.5) * 0.1
        return np.ones((self.l, self.l)) * initialPhi + noise
        
    def laplacian(self, arr: np.ndarray):
        """Computes the laplacian for a 2 dimensional array"""
        return (np.roll(arr, 1, axis=0) + np.roll(arr, 1, axis=1) + \
        np.roll(arr, -1, axis=0) + np.roll(arr, -1, axis=1) - 4 * arr)

    def gradient(self, arr: np.ndarray):
        return (np.roll(arr, -1, axis=0) + np.roll(arr, -1, axis=1) - 2 * arr)

    def compute_mu(self, a=0.1, k=0.1, dx=1):
        self.mu = -a * self.phi + a * self.phi**3 - k*self.laplacian(self.phi)/dx**2

    def compute_phi(self, M=0.1, dt=1, dx=1):
        self.phi = self.phi + M*dt*self.laplacian(self.mu)/dx**2

    def get_free_energy_density(self, a=0.1, k=0.1):
        return -a*self.phi**2 / 2 + a*self.phi**4 / 4 + k*(self.gradient(self.phi))**2 / 2

    def animate_cahn_hilliard(self):
        plt.imshow(self.phi, animated=True, vmin=-1, vmax=1)
        plt.colorbar()
        times = []
        free_energy_density = []
        for epoch in range(self.nstep):
            self.compute_phi()
            self.compute_mu()
            if epoch % 100 == 0:
                times.append(epoch)
                free_energy_density.append(np.sum(self.get_free_energy_density()))
                print(f"Epoch number {epoch}")
                print(np.sum(self.phi))
                # plt.cla()
                # plt.imshow(self.phi, animated=True, vmin=-1, vmax=1)
                # plt.draw()
                # plt.pause(0.0001)

        print(times)
        print(free_energy_density)

        data = {'time': times, 'f': free_energy_density}
        pd.DataFrame.from_dict(data).to_csv('free_energy_data.csv'+f'{self.initialPhi}', index=False)
