from ElectricPoisson import ElectricPoisson
from MagneticPoisson import MagneticPoisson
from algorithms import GaussSeidelUpdate3d, JacobiUpdate2d, overRelaxation
from cahn_hilliard import CahnHilliardSolver

def exp1():
    pde = CahnHilliardSolver(l=50)
    pde.animate_cahn_hilliard()
    pde = CahnHilliardSolver(l=50, initialPhi=0.5)
    pde.animate_cahn_hilliard()

def exp2():
    solver = MagneticPoisson(l=50)
    solver.solvePoisson(update=JacobiUpdate2d)
    solver.plotBfield2d()

def exp3():
    solver = ElectricPoisson(l=50)
    solver.solvePoisson(update=overRelaxation, w=1.95)
    # solver.solvePoisson(update=GaussSeidelUpdate3d)
    # print(solver.potential)
    solver.plotEfield()

def exp4():
    solver = MagneticPoisson(l=50, is2d=False)
    solver.solvePoisson(update=GaussSeidelUpdate3d)
    solver.plotBfield3d()

def exp5():
    solver = ElectricPoisson(l=10)
    # solver.solvePoisson(update=overRelaxation, w=1.99)
    solver.findMinimumW()

if __name__ == '__main__':
    exp5()
    # exp1()
