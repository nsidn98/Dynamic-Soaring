# imports
import pickle
from typing import List
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams.update({'font.size': 16})
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

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
    def __init__(self, travelling:bool=True):
        self.travelling = travelling
        # define the parameters used for simulating stuff
        self.g = 9.8            # acceleration due to gravity in m/s^2
        self.W0 = 7.8           # wind speed at z=10m
        self.delta = 12       # shear layer thickness in m
        self.rho = 1.2          # wind density in kg/m^3
        self.m = 9.5            # mass of albatross in kg
        self.cD_0 = 0.01        # drag coeff
        self.S = 0.65           # wing area in m^2
        # (using value from https://www.dropbox.com/s/ad2j2q9cekpgxic/6832-dynamic-soaring.pdf)
        E_max = 40
        self.k = 1/(4 * self.cD_0 * E_max**2)
    
    def reset(self, q0:List=[0, 0, 0, 10, np.pi/8, 0], tF:int=10, 
                    Nt:int=20, plot:bool=True):
        r"""
            Initialise the albatross at `q0` and simulate a trajectory
            with cL = 1.5 and φ = π/8 for `tF` seconds with `Nt` number
            of discretisation points
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
            Parameter:
            ––––––––––
            z: float or np.ndarray
            ––––––––––
        """
        return self.W0/(1+np.exp(-z/self.delta))

    def windDot(self, z, zdot):
        r"""
            Parameter:
            ––––––––––
            z: float or np.ndarray
            zdot: float or np.ndarray
            ––––––––––
            Evaluate Wdot = (δW/δz).zdot
            NOTE: is using the sigmoid wind profile
        """
        zexp = np.exp(-z/self.delta)
        Wdot = (self.W0/self.delta) * (zexp/(1+zexp)**2) * zdot
        return Wdot
        
    def soaringDynamics(self, q, t, phiOptInterp, cLOptInterp, tF):
        """
            The dynamics equations as defined in Bousquet et al. (2017)
            Parameter:
            ––––––––––
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
            ––––––––––
        """
        # handle the edge case for interpolator
        if t>tF:
            t = tF
            
        phi = phiOptInterp(t)
        cL = cLOptInterp(t)
        

        qdot = self.getQDot([q, phi, cL], vec=False)
        
        return qdot
        
    def getQDot(self, X, vec):
        """
            Given the current state `q` and inputs `phi` and `cL`
            get the `q_dot` vector in accordance to the dynamics
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
        gammadot = (L * np.cos(phi) - self.m * self.g * np.cos(gamma)- self.m * Wdot * np.sin(gamma) * np.sin(psi))/(self.m * V)
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
            sol: np.ndarray
                shape: [num_steps, 3]
            quiver: bool
                If we want to plot the wind quiver
            double: bool
                If we want to plot two time periods
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
        zmin = -10; zmax = max(zline) + k
        verts = [(xmin,ymin,zmin), (xmin,ymax,zmin), (xmax,ymax,zmin), (xmax,ymin,zmin)]
        ax.add_collection3d(Poly3DCollection([verts],color='royalblue',alpha=0.3))
        # ax.view_init(elev=-20, azim=60)
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
        q, _, _, _, _, _, _, _, _ = unpackState(x, self.Nt)
        qdot = self.getQDot(x, vec=True)

        qnext = qdot[:-1,:]*self.dt + q[:-1,:]
        return (qnext - q[1:,:]).flatten()

    def addConstraints(self, Nt):
        r"""
            –––––––––––––––––––––
            Periodic Constraints:
            –––––––––––––––––––––
            • V_Nt = V_0
            • ψ_Nt = ψ_0 + 2π 
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
            • |ψ_i| < 3π
                OR
            • |ψ_i| < π     (for non-circular trajectory)
            • |γ_i| < π/2
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
        if self.travelling:
            #                 x,        y,    z,      V,   ψ,     γ
            lb = np.array([-np.inf, -np.inf, -10,     0, -pi, -pi/2])
            ub = np.array([ np.inf,  np.inf, np.inf, 50,  pi,  pi/2])
        else:
            #                 x,        y,    z,      V,   ψ,     γ
            lb = np.array([-np.inf, -np.inf, -10,     0, -3*pi, -pi/2])
            ub = np.array([ np.inf,  np.inf, np.inf, 50,  3*pi,  pi/2])
        lb = lb.reshape((-1,lb.size)).repeat(Nt,0)
        ub = ub.reshape((-1,ub.size)).repeat(Nt,0)
        lb = lb.T.flatten()
        ub = ub.T.flatten()
        # add bounds for cL and φ
        lb = np.hstack((lb, np.zeros(Nt-1),  -pi*np.ones(Nt-1)))
        ub = np.hstack((ub, 10*np.ones(Nt-1), pi*np.ones(Nt-1)))
        #################################################

        return periodicConstraint, initConstraint, lb, ub
    
    def optimise(self, Nt):
        """
            Perform the optimisation using the solution obtained
            by simulating the trjacetory from the init_state `q0`
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
        dynConstr = NonlinearConstraint(self.dynamicsConstraints, -1e-5, 1e-5)
        traj_opt_sol = minimize(self.cost, x0, method='trust-constr',
                                bounds=Bounds(lb,ub),
                                constraints=[dynConstr, periodicConstr, initConstr],
                                options={'maxiter':200, 'verbose':3,'gtol':1e-4}, 
                                tol=1e-5)

        q_sol, _, _, _, _, _, _, cL_sol, phi_sol = unpackState(traj_opt_sol.x, Nt)
        self.cLBaseInterp = interp1d(tSim, np.append(cL_sol, cL_sol[-1]))
        self.phiBaseInterp = interp1d(tSim, np.append(phi_sol, phi_sol[-1]))
        self.qBaseInterp = interp1d(tSim, q_sol, axis=0)

        self.plot_traj(q_sol, True)

        return traj_opt_sol
    
    def save_traj(self, solution, name:str='traj_opt.pkl'):
        """
            Save the solution obtained
        """
        with open(name, 'wb') as output:
            pickle.dump(solution, output, pickle.HIGHEST_PROTOCOL)

    def load_traj(self, name:str):
        """
            Load the optimised trajectory 
            which was saved using `save_traj`
        """
        with open(name, 'rb') as input:
            traj_opt = pickle.load(input)
        return traj_opt



if __name__ == "__main__":
    t = Albatross(travelling=True)
    Nt = 50
    tF = 7
    t.reset(q0=[0, 0, 0, 10, -np.pi, 0], tF=tF, Nt=Nt, plot=False)
    sol = t.optimise(Nt=Nt)
    t.save_traj(sol, 'sol.pkl')
