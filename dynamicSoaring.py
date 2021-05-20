"""
    Code for dynamic soaring
    Authors: 
        Antonia Bronars @tonibronars
        Rebecca Jiang @rhjiang
        Siddharth Nayak @nsidn98
    Usage:
    python -W ignore dynamicSoaring.py

    This will run all the experiments from the paper
    Refer plot.py for the other plots and stuff related to 
    plotting other misc stuff
"""
# imports
from typing import List, Dict

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams.update({'font.size': 16})
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
sns.set()

from scipy import linalg as la
import random

import control 
import sympy as sym
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, Bounds
from scipy.interpolate import interp1d
from scipy.integrate import odeint

from math import pi

def unpackState(Z, Nt):
        """
            Z is a numpy array of shape (Nt*8)
            where 
                the first Nt terms are [x_0, ... x_Nt]
                the next Nt terms are [y_0, ... y_Nt]
                and so on.
            q = [x, y, z, V, psi, gamma, cL, phi]
        """
        x     = Z[0:Nt]
        y     = Z[Nt:2*Nt]
        z     = Z[2*Nt:3*Nt]
        V     = Z[3*Nt:4*Nt]
        psi   = Z[4*Nt:5*Nt]
        gamma = Z[5*Nt:6*Nt]
        cL    = Z[6*Nt:7*Nt-1]
        phi   = Z[7*Nt-1:]

        q = np.vstack((x, y, z, V, psi, gamma)).T

        return q, x, y, z, V, psi, gamma, cL, phi

class Albatross:
    def __init__(self, travelling:bool=True, m:float=9.5, 
                delta:float=12, cD_0:float=0.01, S:float=0.65, 
                rho:float=1.2, W0:float=7.8, tF:int=10):
        """
            –––––––––––
            Parameters:
            –––––––––––
            travelling: bool
                Whether we want non-circular travelling 
                trajectories or circular (type) trajectories
            m: float
                mass of albatross in kg
            delta: float
                wind shear layer thickness in m
            cD_0: float
                drag coefficient at zero speed
            S: float
                wing area in m^2
            rho: float
                air density in kg/m^3
            W0: float
                wind speed at z=10m
            tF: int
                Time period for trajectory
        """
        self.travelling = travelling
        self.g = 9.8            # acceleration due to gravity in m/s^2
        self.W0 = W0
        self.delta = delta
        self.rho = rho
        self.m = m
        self.cD_0 = cD_0
        self.S = S
        E_max = 40
        self.k = 1/(4 * self.cD_0 * E_max**2) # cD = cD_0 + k*cL**2
        self.sea_level = -10    # sea-level is at z=-10
        self.tF = tF
        #################################################
        # Add functional definitions of linearized A and B matrices 
        # as self.fA and self.fB
        self.linearize()
        self.Q = np.eye(6)
        self.R = 0.01*np.eye(2)
        self.Qf = 15*np.eye(6)
        # noise standard deviations
        self.stdNoise = 1*np.array([0.1,0.1,0.1,1,np.pi/20,np.pi/20])
    
    def reset(self, q0:List=[0, 0, 0, 10, np.pi/8, 0], tF:int=10, 
                    Nt:int=20, plot:bool=True):
        r"""
            Initialise the albatross at `q0` and simulate a trajectory
            with cL = 1.5 and φ = π/8 for `tF` seconds with `Nt` number
            of discretisation points
            –––––––––––
            Parameters:
            –––––––––––
            q0: List
                A list for initialising the albatross
                [x, y, z, V, psi, gamma]
                Default: [0, 0, 0, 10, np.pi/8, 0]
            tF: int
                Time for which we simulate the trajectory
                (Can be treated as the time-period of flight loops)
                Default: 10
            Nt: int
                Number of discretisation points
                Default: 20
            plot: bool
                Whether to plot the simulated trajectory obtained
                Default: True
            ––––––––––
        """
        self.Nt = Nt
        self.tF = tF
        self.dt = tF/Nt
        tSim = np.linspace(0,tF,Nt)
        phiBase = np.pi/8*np.ones((Nt))
        cLBase = 1.5*np.ones((Nt))
        tInterp = tSim
        # For initializing the optimization for the first time
        self.phiBaseInterp = interp1d(tSim, phiBase)
        self.cLBaseInterp = interp1d(tSim, cLBase)
        sol = odeint(self.soaringDynamics, q0, tSim, 
                    args = (self.phiBaseInterp, self.cLBaseInterp, tF))
        
        self.qBaseInterp = interp1d(tSim, sol, axis=0)

        if plot:
            self.plot_traj(sol,True)

    def windProfile(self, z):
        """
            The wind gradient profile as a sigmoid
            –––––––––––
            Parameter:
            –––––––––––
            z: float or np.ndarray
            –––––––––––
        """
        return self.W0/(1+np.exp(-z/self.delta))

    def windDot(self, z, zdot):
        """
            –––––––––––
            Parameter:
            –––––––––––
            z: float or np.ndarray
            zdot: float or np.ndarray
            –––––––––––
            Evaluate Wdot = (δW/δz).zdot
            NOTE: is using the sigmoid wind profile
        """
        zexp = np.exp(-z/self.delta)
        Wdot = (self.W0/self.delta) * (zexp/(1+zexp)**2) * zdot
        return Wdot
        
    def soaringDynamics(self, q, t, phiOptInterp, cLOptInterp, tF):
        """
            The dynamics equations as defined in Bousquet et al. (2017)
            –––––––––––
            Parameter:
            –––––––––––
            q: np.ndarray or List
                The state of the glider
                q = [x, y, z, V, psi, gamma]
            t: float
                Time at which we want to evaluate the controls using an interpolator
            phiOptInterp: interp1d
                Interpolator for phi
            cLOptInterp: interp1d
                Interpolator for cL
            tF: int
                Time for which we simulate the trajectory
                (Can be treated as the time-period of flight loops)
            –––––––––––
        """
        # handle the edge case for interpolator
        if t>tF:
            t = tF
            
        phi = phiOptInterp(t)
        cL = cLOptInterp(t)
        
        qdot = self.getQDot([q, phi, cL], vec=False)
        
        return qdot
        
    def getQDot(self, X, vec:bool):
        """
            Given the current state `q` and inputs `phi` and `cL`
            get the `q_dot` vector in accordance to the dynamics
            –––––––––––
            Parameters:
            –––––––––––
            X: np.ndarray or List
                The state+control
            vec: bool
                If vectorised form or not
            –––––––––––
        """
        if vec:
            _, _, _, z, V, psi, gamma, cL, phi = unpackState(X, self.Nt)

            # put zero at end of sequence because we will throw out the last qdot anyways
            cL = np.append(cL,0)
            phi = np.append(phi,0)

        else:
            q, phi, cL = X
            _, _, z, V, psi, gamma = q
        
        # velocity equations
        zdot = V * np.sin(gamma)
        # wind profile equations
        W = self.windProfile(z)
        Wdot = self.windDot(z, zdot)

        xdot = V * np.cos(gamma) * np.cos(psi)
        ydot = V * np.cos(gamma) * np.sin(psi) - W
        
        cD = self.cD_0 + self.k * (cL **2)
        D = 0.5 * cD * self.rho * self.S * (V**2)
        L = 0.5 * cL * self.rho * self.S * (V**2)
        Vdot = -D/self.m - self.g * np.sin(gamma) + Wdot * np.cos(gamma) * np.sin(psi)
        gammadot = (L * np.cos(phi) - self.m * self.g * np.cos(gamma) - self.m * Wdot * np.sin(gamma) * np.sin(psi))/(self.m * V)
        psidot =  (L * np.sin(phi) + self.m * Wdot + np.cos(psi)) / (self.m * V * np.cos(gamma))

        if vec:
            qdot = np.vstack([xdot, ydot, zdot, Vdot, psidot, gammadot]).T  # shape (Nt, 6)
        else:
            qdot = [xdot, ydot, zdot, Vdot, psidot, gammadot]

        return qdot

    def plot_traj(self, sol:np.ndarray, quiver:bool=False, double:bool=False):
        """
            Plot the trajectory, sea-level plane and the 
            wind-profile vector field
            –––––––––––
            Parameters:
            –––––––––––
            sol: np.ndarray
                shape: [num_steps, 3]
            quiver: bool
                If we want to plot the wind quiver
            double: bool
                If we want to plot two time periods
                by stitching one time period twice
            –––––––––––
        """
        # https://stackoverflow.com/questions/36737053/mplot3d-fill-between-extends-over-axis-limits
        # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')

        # if we want to plot two time periods
        if double:
            sol_copy = sol.copy()
            start_point = sol[0, :2]
            end_point = sol[-1, :2]
            sol_copy[:, :2] = sol_copy[:, :2] + end_point - start_point
            sol = np.concatenate((sol, sol_copy), 0)

        # Data for 3D trajectory
        zline = sol[:,2]
        xline = sol[:,0]
        yline = sol[:,1]
        ax.plot3D(xline, yline, zline, 'gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #####

        # add vertices for sea-level plane
        k = 0
        xmin = min(xline)-k; ymin = min(yline)-k
        xmax = max(xline)+k; ymax = max(yline)+k
        zmin = self.sea_level; zmax = max(zline) + k
        verts = [(xmin,ymin,zmin), (xmin,ymax,zmin), (xmax,ymax,zmin), (xmax,ymin,zmin)]
        ax.add_collection3d(Poly3DCollection([verts],color='royalblue',alpha=0.3))
        #####
        # plot wind profile
        if quiver:
            x_q, y_q, z_q = np.meshgrid(np.arange(xmin, xmax, 5),
                                np.arange(ymin, ymax, 5),
                                np.arange(zmin, zmax, 5))
            u = 0 * x_q
            v = - self.windProfile(z_q)
            w = 0 * x_q
            ax.quiver(x_q, y_q, z_q, u, v, w, length=0.5, color = 'black', alpha=0.3)
        #####
        plt.show()

    def cost(self, x):
        """
            Not minimizing anything so a dummy cost
        """
        return 1

    def dynamicsConstraints(self, x):
        """
            Define constraints imposed due 
            to the dynamics of the albatross
            –––––––––––
            Parameter:
            –––––––––––
            x: np.ndarray
                State + control
            –––––––––––
        """
        q, _, _, _, _, _, _, _, _ = unpackState(x, self.Nt)
        qdot = self.getQDot(x, vec=True)

        qnext = qdot[:-1,:]*self.dt + q[:-1,:]
        return (qnext - q[1:,:]).flatten()

    def addConstraints(self, Nt:int):
        r"""
            –––––––––––––––––––––
            Periodic Constraints:
            –––––––––––––––––––––
            • V_Nt = V_0
            • ψ_Nt = ψ_0 + 2π (for circular trajectory only)
                OR 
            • ψ_Nt = ψ_0    (for non-circular trajectory)
            • γ_Nt = γ_0
            • x_Nt = x_0    (for circular trajectory only)
            • z_Nt = z_0
            –––––––––––––––––––––
            –––––––––––––––––––––
            Init Constraint:
            –––––––––––––––––––––
            • z_0 = 0
            –––––––––––––––––––––
            –––––––––––––––––––––
            Upper/Lower Bounds:
            –––––––––––––––––––––
            • V_i > 0
            • cL_i > 0
            • |ψ_i| < 3π.   (for circular trajectory only)
                OR
            • |ψ_i| < π     (for non-circular trajectory)
            • |γ_i| < π/2
            –––––––––––––––––––––
        """
        ############# Periodic Constraints #############
        if self.travelling:
            periodicConstraintMatrix = np.zeros((4, Nt*8-2))
        else:
            periodicConstraintMatrix = np.zeros((5, Nt*8-2))
        periodicConstraintMatrix[0, 2*Nt]   = 1    # z_0
        periodicConstraintMatrix[0, 3*Nt-1] = -1   # z_Nt
        periodicConstraintMatrix[1, 3*Nt]   = 1    # V_0
        periodicConstraintMatrix[1, 4*Nt-1] = -1   # V_Nt
        periodicConstraintMatrix[2, 4*Nt]   = 1    # ψ_0
        periodicConstraintMatrix[2, 5*Nt-1] = -1   # ψ_Nt
        periodicConstraintMatrix[3, 5*Nt]   = 1    # γ_0
        periodicConstraintMatrix[3, 6*Nt-1] = -1   # γ_Nt
        if self.travelling:
            lb = np.array([0, 0, 0, 0 ])
            ub = np.array([0, 0, 0, 0 ])
        else:
            periodicConstraintMatrix[4, 0]    = 1    # x_0
            periodicConstraintMatrix[4, Nt-1] = -1   # x_Nt
            lb = np.array([0, 0, -2*pi, 0, 0 ])
            ub = np.array([0, 0, -2*pi, 0, 0 ])

        periodicConstraint = LinearConstraint(periodicConstraintMatrix, lb, ub)
        #################################################

        ############# Init Constraints #############
        initConstraintMatrix = np.zeros((1, 8*Nt-2))
        initConstraintMatrix[0, 2*Nt] = 1     # z_0

        initConstraint = LinearConstraint(initConstraintMatrix, 0, 0)
        #################################################

        ############# Upper/Lower Bounds #############
        zmin = self.sea_level
        if self.travelling:
            #                 x,        y,    z,      V,   ψ,     γ
            lb = np.array([-np.inf, -np.inf, zmin,    0, -pi, -pi/2])
            ub = np.array([ np.inf,  np.inf, np.inf, 50,  pi,  pi/2])
        else:
            #                 x,        y,    z,      V,   ψ,     γ
            lb = np.array([-np.inf, -np.inf, zmin,    0, -3*pi, -pi/2])
            ub = np.array([ np.inf,  np.inf, np.inf, 50,  3*pi,  pi/2])
        lb = lb.reshape((-1,lb.size)).repeat(Nt,0)
        ub = ub.reshape((-1,ub.size)).repeat(Nt,0)
        lb = lb.T.flatten()
        ub = ub.T.flatten()
        # add bounds for cL and φ
        lb = np.hstack((lb, np.zeros(Nt-1),  -pi/2*np.ones(Nt-1)))
        ub = np.hstack((ub, 10*np.ones(Nt-1), pi/2*np.ones(Nt-1)))
        #################################################

        return periodicConstraint, initConstraint, lb, ub
    
    def optimise(self, Nt:int, plot:bool=False, verbose_level:int=3, maxiter:int=200):
        """
            Perform the trajectory optimisation
            by collecting all the constraints,
            the cost function and the initial guess
            –––––––––––
            Parameters:
            –––––––––––
            Nt: int
                The number of collocation points in the trajectory optimisation
            plot: bool
                Whether to plot the obtained trajectory
            verbose_level: int
                Can be one of {0, 1, 2, 3}
            max_iter: int
                Number of iterations to run the optimisation algorithm
            –––––––––––
        """
        self.Nt = Nt
        self.dt = self.tF/Nt

        tSim = np.linspace(0, self.tF, Nt)
        # interpolate the baseline guess to the current time mesh
        phi0 = self.phiBaseInterp(tSim[:-1])
        cL0 = self.cLBaseInterp(tSim[:-1])
        q0 = self.qBaseInterp(tSim) # [timesteps x 6]
        x0 = q0.T.flatten()
        x0 = np.hstack((x0, cL0, phi0))

        periodicConstr, initConstr, lb, ub = self.addConstraints(self.Nt)
        dynConstr = NonlinearConstraint(self.dynamicsConstraints, 0, 0)
        traj_opt_sol = minimize(self.cost, x0, method='trust-constr',
                                bounds=Bounds(lb,ub),
                                constraints=[dynConstr, periodicConstr, initConstr],
                                options={'maxiter':maxiter, 'verbose':verbose_level,'gtol':1e-4}, 
                                tol=1e-5)

        q_sol, _, _, _, _, _, _, cL_sol, phi_sol = unpackState(traj_opt_sol.x, Nt)
        self.cLBaseInterp = interp1d(tSim, np.append(cL_sol, cL_sol[-1]))
        self.phiBaseInterp = interp1d(tSim, np.append(phi_sol, phi_sol[-1]))
        self.qBaseInterp = interp1d(tSim, q_sol, axis=0)

        if plot:
            self.plot_traj(q_sol, True)

        return traj_opt_sol
    
    def save_traj(self, solution, name:str='traj_opt.pkl'):
        """
            Save the solution obtained
            –––––––––––
            Parameters:
            –––––––––––
            solution: object
                The scipy solution object to save
            name: str
                The file name with which to store the file
            –––––––––––
        """
        with open(name, 'wb') as output:
            pickle.dump(solution, output, pickle.HIGHEST_PROTOCOL)

    def load_traj(self, name:str, Nt:int, tF:int=7):
        """
            Load the optimised trajectory 
            which was saved using `save_traj`
            Will also load interpolators with the solution 
            –––––––––––
            Parameters:
            –––––––––––
            name: str
                The file name with which to store the file
            Nt: int
                The number of collocation points in the trajectory optimisation
            tF: int
                The time period of the trajectory
            –––––––––––
        """
        with open(name, 'rb') as input:
            solution = pickle.load(input)
        #############################
        # load the interpolators with the solution loaded
        tSim = np.linspace(0, tF, Nt)
        q_sol, _, _, _, _, _, _, cL_sol, phi_sol = unpackState(solution.x, Nt)
        self.cLBaseInterp = interp1d(tSim, np.append(cL_sol, cL_sol[-1]))
        self.phiBaseInterp = interp1d(tSim, np.append(phi_sol, phi_sol[-1]))
        self.qBaseInterp = interp1d(tSim, q_sol, axis=0)
        #############################
        return solution
    
    def linearize(self):
        """
            Linearise the dynamics equation
            of the albatross flight symbolically
        """
        # define symbolic variables
        V = sym.Symbol('V',real=True)
        gamma = sym.Symbol('gamma',real=True)
        psi = sym.Symbol('psi',real=True)
        x = sym.Symbol('x',real=True)
        y = sym.Symbol('y',real=True)
        z = sym.Symbol('z',real=True)
        phi = sym.Symbol('phi',real=True)
        cL = sym.Symbol('cL',real=True)
        ############################

        # define dynamics equations in symbolic form
        zdot = V * sym.sin(gamma)

        zexp = sym.exp(-z/self.delta)
        W = self.W0/(1+zexp)
        Wdot = (self.W0/self.delta) * (zexp/(1+zexp)**2) * zdot

        xdot = V * sym.cos(gamma) * sym.cos(psi)
        ydot = V * sym.cos(gamma) * sym.sin(psi) - W

        cD = self.cD_0 + self.k * (cL **2)
        D = 0.5 * cD * self.rho * self.S * (V**2)
        L = 0.5 * cL * self.rho * self.S * (V**2)
        Vdot = -D/self.m - self.g * sym.sin(gamma) + Wdot * sym.cos(gamma) * sym.sin(psi)
        gammadot = (L * sym.cos(phi) - self.m * self.g * sym.cos(gamma) - self.m * Wdot * sym.sin(gamma) * sym.sin(psi))/(self.m * V)
        psidot = (L * sym.sin(phi) + self.m * Wdot + sym.cos(psi)) / (self.m * V * sym.cos(gamma))

        qdot = sym.Matrix([xdot, ydot, zdot, Vdot, psidot, gammadot])

        q = sym.Matrix([x,y,z,V,psi,gamma])
        u = sym.Matrix([cL,phi])
        ############################

        # define linearizing matrices
        A = qdot.jacobian(q)
        B = qdot.jacobian(u)

        self.fA = sym.lambdify(((x, y, z, V, psi, gamma, cL, phi),), A, "numpy")
        self.fB = sym.lambdify(((x, y, z, V, psi, gamma, cL, phi),), B, "numpy")
        ############################

    def lqrGainsInfinite(self, qnom:np.ndarray, cL, phi):
        """
            Calculate the LQR Gains around the 
            nominal trajectory with infinite horizon
            –––––––––––
            Parameters:
            –––––––––––
            qnom: np.ndarray
                shape [Nt, 6]
                The nominal trajectory
            cL: np.ndarray
                shape [Nt, 1]
            phi: np.ndarray
                shape [Nt,]
            –––––––––––
        """
        tSim = np.linspace(0, self.tF, self.Nt)
        K_all = np.zeros((self.Nt, 2, 6))
        K_ss = np.zeros((self.Nt, 2, 6))

        # evaluate A[t] and B[t] and the gains K[t]
        for i in range(len(tSim)-1):
            x, y, z, V, psi, gamma = qnom[i]
            A = self.fA((x, y, z, V, psi, gamma, cL[i], phi[i]))
            B = self.fB((x, y, z, V, psi, gamma, cL[i], phi[i]))
            K, _, _ = control.lqr(A, B, self.Q, self.R)
            K_all[i,:,:] = K
        K_all[-1,:,:] = K_all[-2,:,:]   # add last time step Ks

        # store the gains in an interpolator
        self.KInterp = interp1d(tSim, K_all, axis=0)
        self.q0Interp = interp1d(tSim, qnom, axis=0)
        self.cL0Interp = interp1d(tSim, np.append(cL, cL[-1]))
        self.phi0Interp = interp1d(tSim, np.append(phi, phi[-1]))

    def lqrGainsFinite(self, qnom, cL, phi, noise):
        """
            Calculate the LQR Gains around the 
            nominal trajectory with finite horizon
            Solves the Continuous Algebraic Riccati Equation (CARE)
            internally and calculates the gains at the collocation points 
            and then stores them in an interpolator for future use
            –––––––––––
            Parameters:
            –––––––––––
            qnom: np.ndarray
                shape [Nt, 6]
                The nominal trajectory
            cL: np.ndarray
                shape [Nt, 1]
            phi: np.ndarray
                shape [Nt,]
            noise: np.ndarray
                shape [Nt, 6]
            –––––––––––
        """
        tSim = np.linspace(0, self.tF, self.Nt)

        # make interpolators for q, cL, phi and noise
        self.q0Interp = interp1d(tSim, qnom, axis=0)
        self.cL0Interp = interp1d(tSim, np.append(cL, cL[-1]))
        self.phi0Interp = interp1d(tSim, np.append(phi, phi[-1]))
        self.noiseInterp = interp1d(tSim, noise, axis=0)

        i = len(tSim) - 2
        x, y, z, V, psi, gamma = qnom[i]
        A = self.fA((x,y,z,V,psi,gamma,cL[i],phi[i]))
        B = self.fB((x,y,z,V,psi,gamma,cL[i],phi[i]))
        Pss = self.Qf
        Nx = A.shape[0]

        P = odeint(self.Pdot, Pss.reshape(Nx**2,), tSim)
        K = np.zeros((2,Nx,len(tSim)))
        Pmat = np.zeros((Nx,Nx,len(tSim)))
        # Solve CARE for each time step in the 
        # nominal trajectory and calculate the gains
        for i in np.arange(0,len(tSim)):
            t = tSim[self.Nt-1-i]
            x, y, z, V, psi, gamma = qnom[i]
            phi = self.phi0Interp(t)
            cL = self.cL0Interp(t)
            B = self.fB((x,y,z,V,psi,gamma,cL,phi))
            K[:,:,self.Nt-1-i] = np.linalg.inv(self.R) @ B.T @ (P[i,:].reshape(Nx,Nx))
            Pmat[:,:,self.Nt-1-i] = P[i,:].reshape(Nx,Nx)

        # store the gains in an interpolator for future use
        self.KInterp = interp1d(tSim, K, axis=2)

    def Pdot(self, P, t):
        """
            Use this as an input for the odeint
            for calculating the P matrix in the
            Continuous Algebraic Riccati Equation (CARE)
        """
        if t>self.tF:
            t = self.tF

        phi = self.phi0Interp(t)
        cL = self.cL0Interp(t)
        qnom = self.q0Interp(t)

        x, y, z, V, psi, gamma = qnom
        A = self.fA((x,y,z,V,psi,gamma,cL,phi))
        B = self.fB((x,y,z,V,psi,gamma,cL,phi))

        Nx = A.shape[0]
        P = P.reshape(Nx, Nx)
        P = (P + P.T) / 2
        dotP = P @ A + A.T @ P + self.Q - P @ B @ np.linalg.inv(self.R) @ B.T @ P
        return dotP.reshape(Nx*Nx,)

    def lqrDynamics(self, q, t, noise:bool=True):
        """
            Dynamics, under lqr control using finite-horizon (steady-state) gains:
            –––––––––––
            Parameters:
            –––––––––––
            q: np.ndarray or List
                The state of the glider
                q = [x, y, z, V, psi, gamma]
            t: float
                Time at which we want to evaluate the controls using an interpolator
            noise: bool
                If we want to add noise to the nominal trajectories
            –––––––––––
        """
        # handle the edge case for interpolator
        q = np.array(q)

        if t>self.tF:
            t = self.tF
            
        phi0 = self.phi0Interp(t)
        cL0 = self.cL0Interp(t)
        q0 = self.q0Interp(t)
        K = self.KInterp(t)

        uBar =  -K@(q-q0)
        cL = cL0 + uBar[0]
        phi = phi0 + uBar[1]

        n = self.noiseInterp(t)

        qdot = self.getQDot([q.tolist(), phi, cL], vec=False)
        if noise:
            qdot = (np.array(qdot) + n).tolist()
        return qdot

    def simulateLqr(self, noise:bool=False, plot:bool=True):
        """
            Simulate the LQR experiment
            –––––––––––
            Parameters:
            –––––––––––
            noise: bool
                If we want to add noise to the nominal trajectory
            plot: bool
                If we want to plot the trajectory obtained
            –––––––––––
        """
        tSim = np.linspace(0,self.tF,self.Nt)
        q0 = self.q0Interp(0).tolist()
        sol = odeint(self.lqrDynamics, q0, tSim)
        if plot:
            self.plot_traj(sol,True)
        return sol

    def runSolOpenLoop(self, plot:bool=False):
        """
            Simulate the LQR experiment
            –––––––––––
            Parameters:
            –––––––––––
            plot: bool
                If we want to plot the trajectory obtained
        """
        tSim = np.linspace(0,self.tF,self.Nt)
        q0 = self.q0Interp(0).tolist()
        sol = odeint(self.openLoopDynamics, q0, tSim)
        if plot:
            self.plot_traj(sol,True)
        return sol

    def openLoopDynamics(self, q, t, noise:bool=True):
        """
            Dynamics, under open loop control
            –––––––––––
            Parameters:
            –––––––––––
            q: np.ndarray or List
                The state of the glider
                q = [x, y, z, V, psi, gamma]
            t: float
                Time at which we want to evaluate the controls using an interpolator
            noise: bool
                If we want to add noise to the nominal trajectory
            –––––––––––
        """
        # handle the edge case for interpolator
        q = np.array(q)

        if t>self.tF:
            t = self.tF
            
        phi0 = self.phi0Interp(t)
        cL0 = self.cL0Interp(t)

        qdot = self.getQDot([q.tolist(), phi0, cL0], vec=False)

        n = self.noiseInterp(t)
        if noise:
            qdot = (np.array(qdot) + n).tolist()

        return qdot
    
    def plotLQR(self, lqr_sol:np.ndarray, q_sol:np.ndarray, 
                ol_sol:np.ndarray, tSim:np.ndarray):
        """
            Plot the LQR plots
            –––––––––––
            Parameters:
            –––––––––––
            lqr_sol: np.ndarray
                Shape [Nt, 6]
                Solution from LQR stabilisation
            q_sol: np.ndarray
                Shape [Nt, 6]
                Solution from trajectory optimisation (nominal)
            ol_sol: np.ndarray
                Shape [Nt, 6]
                Solution from Open Loop stabilisation
            tSim: np.ndarray
                Shape: [Nt,]
            –––––––––––
        """
        fig, ax = plt.subplots(3,2, figsize=(15,15))
        fig.tight_layout()
        i = 0
        plt.rc('axes', labelsize=15)
        ylabels = [r"$x\ (m)$", r"$y\ (m)$", r"$z\ (m)$", 
                    r"$V\ (m/s)$", r"$\psi\ (rad)$", r"$\gamma\ (rad)$"]

        for row in range(3):
            for col in range(2):
                ax[row,col].plot(tSim,lqr_sol[:,i],'g', label='LQR')
                ax[row,col].plot(tSim,q_sol[:,i],'r', label='Nominal')
                ax[row,col].plot(tSim,ol_sol[:,i],'b', label='Open Loop')
                ax[row,col].set_ylabel(ylabels[i],fontsize=20)
                i = i + 1
                
                if i==2:
                    ax[row, col].legend(prop={'size': 20})
                if row == 2:
                    ax[row,col].set_xlabel('t (s)',fontsize=20)


def run_exp(params:Dict, save_name:str, Nt:int=50):
    """
        Run trajectory optimisation experiments with 
        different parameters
        params: dict
            {'travelling':True, 'm':9.5, 'delta':12, 'cD_0':0.01, 
                'S':0.65, 'rho':1.2, 'W0':7.8, 'sigmoid':True, 'tF':7}
        save_name: str
            Name with which to save the optimisation results
            Will save the scipy optimisation object along with params
            as a dict. Use `load_solutions` to load the solution
        Nt: Number of collocation points in the trajectory optimisation
    """
    albatross = Albatross(**params)
    albatross.reset(q0=[0, 0, 0, 10, -np.pi, 0], tF=params['tF'], Nt=Nt, plot=False)
    solution = albatross.optimise(Nt=Nt, maxiter=600, plot=False, verbose_level=1)
    params['Nt'] = Nt
    print('_'*50)
    print('Saving file')
    save_obj = {}
    save_obj['solution'] = solution
    save_obj['params'] = params
    albatross.save_traj(save_obj, save_name)
    if not solution.success:
        print('No success with the following params: ', params)
        print(f'Constraint Violation: {solution.constr_violation}')
        print('_'*50)
    del params['Nt']

def load_solutions(tF:int, travel:bool, W0:float, delta:float):
    """
        To load the solutions saved in `run_exp()`
        Feed in the parameters which you want to load
        The solutions are saved as:
            travel_True_delta_7_W0_7.8_tF_7.pkl
        Choose any param from files saved in 'solutions/' folder
    """
    file_name = f'solutions/travel_{travel}_delta_{delta}_W0_{W0}_tF_{tF}.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as input:
            saved_object = pickle.load(input)
        solution = saved_object['solution']
        params = saved_object['params']
        print('_'*50)
        print(f'Optimisation Success: {solution.success}')
        print('_'*50)
        q, x, y, z, V, psi, gamma, cL, phi = unpackState(solution.x, params['Nt'])
        return q, solution, params
    else:
        print('_'*50)
        print('Solution with the given params are not saved in the folder')
        print('_'*50)


if __name__ == "__main__":
    import os
    if not os.path.exists('solutions/'):
        os.makedirs('solutions/')
    
    params = {'travelling':True, 'm':9.5, 'delta':12, 'cD_0':0.01, 
                'S':0.65, 'rho':1.2, 'W0':7.8, 'sigmoid':True, 'tF':7}

    # run the trajectory optimisation experiments for different param combos
    # NOTE This might take long time so choose params wisely
    travellings = [True, False]
    tFs = [5, 7, 12]
    W0s = [5, 7.8, 12]
    deltas = [7, 12, 3, 1]
    for delta in deltas:
        for W0 in W0s:
            for tF in tFs:
                for travel in travellings:
                    params['travelling'] = travel
                    params['tF'] = tF
                    params['W0'] = W0
                    params['delta'] = delta
                    print('_'*50)
                    print(f'Travel: {str(travel)} Delta: {delta} W0: {W0} tF: {tF}')
                    print('_'*50)
                    save_name = f'solutions/travel_{travel}_delta_{delta}_W0_{W0}_tF_{tF}.pkl'
                    run_exp(params, save_name)
    ############################################

    # run the LQR experiments with default params
    random.seed(5)
    tF, Nt = 7, 50
    tSim = np.linspace(0, tF, Nt)
    albatross = Albatross(travelling = True)
    albatross.reset(q0=[0, 0, 0, 10, -np.pi, 0], tF=tF, Nt=Nt, plot=False)
    # optimise trajectory with default params
    solution = albatross.optimise(Nt=Nt, maxiter=600, plot=False, verbose_level=1)
    q_sol, _, _, _, _, _, _, cL_sol, phi_sol = unpackState(solution.x, Nt)

    # Generate vector of random noise, same shape as q_sol
    noise = np.random.normal(0,albatross.stdNoise,
                            size=(q_sol.shape[0], q_sol.shape[1])).tolist()
    # calculate gains for finite horizon
    albatross.lqrGainsFinite(q_sol, cL_sol, phi_sol, noise)
    albatross.plot_traj(q_sol[:-1],True)
    # simulate LQR with noise added into the system
    lqr_sol = albatross.simulateLqr(noise=True)
    # simulate open-loop with noise added into the system
    ol_sol = albatross.runSolOpenLoop()
    albatross.plotLQR(lqr_sol, q_sol, ol_sol, tSim)