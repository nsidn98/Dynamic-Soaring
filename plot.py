# imports
import os
import pickle
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %matplotlib inline
matplotlib.rcParams.update({'font.size': 16})
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


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

def windProfile(z, W0=7.8, delta=12):
    """
        The wind gradient profile as a sigmoid
        Parameter:
        ––––––––––
        z: float or np.ndarray
        ––––––––––
    """
    return W0/(1+np.exp(-z/delta))

def plot_traj(sol:np.ndarray, quiver:bool=False, double:bool=False,
                travelling:bool=True,
                **params):
    """
        Plot the trajectory, sea-level plane and the 
        wind-profile vector field
        sol: np.ndarray
            shape: [num_steps, 3]
        quiver: bool
            If we want to plot the wind quiver
        double: bool
            If we want to plot two time periods
            by stitching one time period twice
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
        v = - windProfile(z_q, W0=params['W0'], delta=params['delta'])
        w = 0 * x_q
        ax.quiver(x_q, y_q, z_q, u, v, w, length=0.5, color = 'black', alpha=0.3)
    #####
    title = 'Trajectory of albatross over two time periods'
    if travelling:
        title += ' (travelling)'
    else:
        title += ' (non-travelling)'
    plt.title(title)
    plt.show()

def plotEnergy(sol, g=9.8):
    z = sol[:, 2]
    V = sol[:, 3]
    PE = g * (z+10) # adding 10 because z=-10 is sea-level
    KE = 0.5 * V**2
    TE = KE + PE
    plt.plot(PE, label='PE')
    plt.plot(KE, label='KE')
    plt.plot(TE, label='TE')
    plt.xlabel('timesteps')
    plt.ylabel('Energy (J/kg)')
    plt.title('Energy (per kg) variation over a time period')
    plt.legend()
    plt.show()

def plotRoll(roll):
    """
        roll == phi
    """
    plt.plot(np.rad2deg(roll))
    plt.xlabel('timesteps')
    plt.ylabel(r'Roll angle $\phi$')
    plt.title(r'Variation of Roll angle $\phi$')
    plt.show()


def load_solutions(tF, travel, W0, delta):
    """
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


def double_traj(sol):
    sol_copy = sol.copy()
    start_point = sol[0, :2]
    end_point = sol[-1, :2]
    sol_copy[:, :2] = sol_copy[:, :2] + end_point - start_point
    sol = np.concatenate((sol, sol_copy), 0)
    return sol

def plot_mul_traj(sol1:np.ndarray, sol2:np.ndarray, 
                sol3:np.ndarray, sol4:np.ndarray=None,
                quiver:bool=False, double:bool=False,
                travelling:bool=True, title:str='Plot',
                labels:list=['r','g','b'],
                W0=7.8, delta=12):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    # if we want to plot two time periods
    if double:
        sol1 = double(sol1)
        sol2 = double(sol2)
        sol3 = double(sol3)

    # Data for 3D trajectory
    ax.plot3D(sol1[:,0], sol1[:,1], sol1[:,2], 'red', label=labels[0])
    ax.plot3D(sol2[:,0], sol2[:,1], sol2[:,2], 'green', label=labels[1])
    ax.plot3D(sol3[:,0], sol3[:,1], sol3[:,2], 'blue', label=labels[2])
    if sol4 is not None:
        ax.plot3D(sol4[:,0], sol4[:,1], sol4[:,2], 'black', label=labels[3])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #####

    # add vertices for sea-level plane
    k = 0
    xmin = min(min(sol1[:,0]), min(sol2[:,0]), min(sol3[:,0]))
    ymin = min(min(sol1[:,1]), min(sol2[:,1]), min(sol3[:,1]))
    if sol4 is not None:
        xmin = min(xmin, min(sol4[:,0]))
        ymin = min(ymin, min(sol4[:,1]))
    xmin = xmin-k; ymin = ymin-k
    xmax = max(max(sol1[:,0]), max(sol2[:,0]), max(sol3[:,0]))
    ymax = max(max(sol1[:,1]), max(sol2[:,1]), max(sol3[:,1]))
    zmax = max(max(sol1[:,2]), max(sol2[:,2]), max(sol3[:,2]))
    if sol4 is not None:
        xmax = max(xmax, max(sol4[:,0]))
        ymax = max(ymax, max(sol4[:,1]))
        zmax = max(zmax, max(sol4[:,2]))
    xmax = xmax+k; ymax = ymax+k
    zmin = -10; zmax = zmax + k
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
        v = - windProfile(z_q, W0=W0, delta=delta)
        w = 0 * x_q
        ax.quiver(x_q, y_q, z_q, u, v, w, length=0.5, color = 'black', alpha=0.3)
    #####
    # title = 'Trajectory of albatross over two time periods'
    if travelling:
        title += ' (travelling)'
    else:
        title += ' (non-travelling)'
    plt.title(title)
    plt.legend()
    plt.show()

def plotWind(x):
    """
        Function to plot the wind profile diagram
    """
    plt.plot(sigmoid(x,k=1,delta=0.5),x, color='black')
    plt.xlim([-0.5,3])
    xlims_arr = np.linspace(-0.5,3,10)
    plt.plot(xlims_arr,np.zeros(xlims_arr.shape),'blue', linewidth=2)
    plt.plot(xlims_arr,-10*np.ones(xlims_arr.shape),'royalblue', linewidth=20)
    plt.title('Wind layer profile')
    plt.ylabel('z')
    plt.text(0.5, 8, 'Wind', fontsize=12)
    plt.text(0.5, 7, '$W_0m/s$', fontsize=12)
    plt.arrow(0.1,5, 5,0, width=0.1, length_includes_head=True,
        head_width=0.08, head_length=0.00002)
    plt.text(0.2, -8, 'No Wind', fontsize = 12)
    #plt.text(0.5, -1.5, 'Wind Shear', fontsize = 12)
    plt.text(0.5, -3.5, 'Layer', fontsize = 12)
    plt.text(1, -10, 'Sea-surface', fontsize = 12)
    plt.annotate("Wind-Shear", xy=(1, 0), xytext=(0.5, -2.5), arrowprops=dict(arrowstyle="->", color='black'))
    plt.annotate("",xy=(1,6), xytext=(0.5,6),arrowprops=dict(arrowstyle="->", color='black'))
    #plt.xticks([])
    plt.show()

def symbolicLinearisation():
    """
        Linearise the albatross dynamics symbolically 
        and print the A and B matrix in LaTeX
    """
    m = sym.Symbol('m',real=True)
    g = sym.Symbol('g',real=True)
    cD_0 = sym.Symbol('cD_0',real=True)
    k = sym.Symbol('k',real=True)
    rho = sym.Symbol('rho',real=True)
    S = sym.Symbol('S',real=True)
    delta = sym.Symbol('delta',real=True)
    W0 = sym.Symbol('W0',real=True)
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

    zexp = sym.exp(-z/delta)
    W = W0/(1+zexp)
    Wdot = (W0/delta) * (zexp/(1+zexp)**2) * zdot

    xdot = V * sym.cos(gamma) * sym.cos(psi)
    ydot = V * sym.cos(gamma) * sym.sin(psi) - W

    cD = cD_0 + k * (cL **2)
    D = 0.5 * cD * rho * S * (V**2)
    L = 0.5 * cL *rho * S * (V**2)
    Vdot = -D/m - g * sym.sin(gamma) + Wdot * sym.cos(gamma) * sym.sin(psi)
    gammadot = (L * sym.cos(phi) -m * g * sym.cos(gamma) - m * Wdot * sym.sin(gamma) * sym.sin(psi))/(m * V)
    psidot = (L * sym.sin(phi) + m * Wdot + sym.cos(psi)) / (m * V * sym.cos(gamma))

    qdot = sym.Matrix([xdot, ydot, zdot, Vdot, psidot, gammadot])

    q = sym.Matrix([x,y,z,V,psi,gamma])
    u = sym.Matrix([cL,phi])
    ############################

    # define linearizing matrices
    A = qdot.jacobian(q)
    B = qdot.jacobian(u)

    print(sym.latex(A))
    print(sym.latex(B))


if __name__ == "__main__":
    # vary time-period
    # travel, W0, delta = False, 7.8, 7
    # tFs = [5, 7, 12]
    # q1,_,_=load_solutions(tFs[0], travel, W0, delta)
    # q2,_,_=load_solutions(tFs[1], travel, W0, delta)
    # q3,_,_=load_solutions(tFs[2], travel, W0, delta)
    # plot_mul_traj(q1, q2, q3, travelling=travel, 
    #             title='Trajectories with different time periods', 
    #             labels=['5s','7s','12s'], 
    #             quiver=True, W0=W0, delta=delta)
    ######################
    
    # vary wind strength
    # tF, travel, delta = 7, True, 7
    # W0s = [5, 7.8, 12]
    # q1,_,_=load_solutions(tF, travel, W0s[0], delta)
    # q2,_,_=load_solutions(tF, travel, W0s[1], delta)
    # q3,_,_=load_solutions(tF, travel, W0s[2], delta)
    # plot_mul_traj(q1, q2, q3, travelling=travel, 
    #             title=r'Trajectories with different wind strength $W_0$', 
    #             labels=['5','7.8','12'], 
    #             quiver=False)
    ######################

    # vary shear layer thickness
    # tF, travel, W0 = 7, True, 7.8
    # deltas = [1, 3, 7, 12]
    # q1,_,_=load_solutions(tF, travel, W0, deltas[0])
    # q2,_,_=load_solutions(tF, travel, W0, deltas[1])
    # q3,_,_=load_solutions(tF, travel, W0, deltas[2])
    # q4,_,_=load_solutions(tF, travel, W0, deltas[3])
    # plot_mul_traj(q1, q2, q3, q4, travelling=travel, 
    #             title=r'Trajectories with different shear layer thickness $\delta$', 
    #             labels=['1','3','7', '12'], 
    #             quiver=False)
    ######################
    tF, travel, W0, delta = 7, True, 7.8, 7
    q,_,_=load_solutions(tF, travel, W0, delta)
    plot_traj(q, travelling=travel, quiver=False, double=True, **{'W0':W0, 'delta':delta})