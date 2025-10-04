#%%
%matplotlib widget
# %matplotlib inline
from scipy.optimize import fsolve
# from scipy.differentiate import hessian
import numpy as np
from numpy import einsum
import matplotlib
matplotlib.pyplot.close('all')
matplotlib.use('module://matplotlib_inline.backend_inline')  # or 'TkAgg' or 'Qt5Agg'
import matplotlib.pyplot as plt

import sympy as sp
from IPython.display import display, Math
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation

import scipy.io as sio
import numpy as np
from matplotlib import ticker

#%%
##################################################
# Functions
##################################################

def new_prob(num):
    print_string = '\n----------------------\n' + 'Assignment E' + str(num) + '\n----------------------\n'
    return print(print_string)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

# Set the global font sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=(8,4))

dpi = 300

def plotta_subplots(
    nrows=1,
    ncols=1,
    fig_size=None,
    fig_title="Figure",
    subplot_instructions=None,  # List of dicts, one per subplot
    suptitle=None,
    savefig=True,
    show=True
):
    """
    Flexible multi-subplot plotting function.
    Each subplot is described by a dict with keys:
      - plotstyle: 'plot', 'scatter', 'contour', 'contourf', 'surface'
      - x1_data, x2_data, x3_data, color, linestyle, marker, cmap, levels, title, xlabel, ylabel, zlabel, frame, elev, azim
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, squeeze=False, subplot_kw={'projection': None})
    axes = axes.flatten()
    if subplot_instructions is None:
        subplot_instructions = [{} for _ in range(nrows * ncols)]
    for idx, instr in enumerate(subplot_instructions):
        ax = axes[idx]
        plotstyle = instr.get("plotstyle", "plot")
        x1_data = instr.get("x1_data", None)
        x2_data = instr.get("x2_data", None)
        x3_data = instr.get("x3_data", None)
        color = instr.get("color", "b")
        linestyle = instr.get("linestyle", "-")
        marker = instr.get("marker", None)
        cmap = instr.get("cmap", "viridis")
        levels = instr.get("levels", 50)
        title = instr.get("title", f"Subplot {idx+1}")
        xlabel = instr.get("xlabel", "x1")
        ylabel = instr.get("ylabel", "x2")
        zlabel = instr.get("zlabel", "Z")
        frame = instr.get("frame", None)
        elev = instr.get("elev", 30)
        azim = instr.get("azim", 45)

        if plotstyle == "plot":
            ax.plot(x1_data, x2_data, color=color, linestyle=linestyle, marker=marker)
        elif plotstyle == "scatter":
            ax.scatter(x1_data, x2_data, color=color, marker=marker)
        elif plotstyle in ["contour", "contourf"]:
            if x3_data is None:
                raise ValueError("x3_data (Z values) required for contour plots")
            if plotstyle == "contour":
                cp = ax.contour(x1_data, x2_data, x3_data, levels=levels, cmap=cmap)
            else:
                cp = ax.contourf(x1_data, x2_data, x3_data, levels=levels, cmap=cmap)
            fig.colorbar(cp, ax=ax)
        elif plotstyle == "surface":
            # For 3D, need to convert axis to 3D
            ax.remove()
            ax = fig.add_subplot(nrows, ncols, idx+1, projection="3d")
            surf = ax.plot_surface(x1_data, x2_data, x3_data, cmap=cmap)
            ax.set_zlabel(zlabel)
            ax.view_init(elev=elev, azim=azim)
            fig.colorbar(surf, ax=ax)
        else:
            raise ValueError(f"Unknown plotstyle: {plotstyle}")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if frame is not None:
            ax.axis(frame)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_title + ".png")
    if show:
        plt.show()
    return fig, axes

#%%
##################################################
# Given code
##################################################

plt.rcParams.update({'font.size': 22})

viscos=1.52e-5

xc=np.loadtxt("xc.dat")
yc=np.loadtxt("yc.dat")

i_lower_sym=20 # the flat plate starts at i=i_lower_sym+1
# boundary layer flow
data=np.genfromtxt("boundary_layer_data.dat", comments="%")

ni=252  # number of grid nodes in x_1 direction
nj=200  # number of grid nodes in x_2 direction

v1=data[:,0] #don't use this array
v2=data[:,1] #don't use this array
p=data[:,2]  #don't use this array

# transform the arrays from 1D fields x(n) to 2D fields x(i,j)
# the first index 'i', correponds to the x-direction
# the second index 'j', correponds to the y-direction

v1_2d=np.reshape(v1,(ni,nj)) #this is v_1 (streamwise velocity component)
v2_2d=np.reshape(v2,(ni,nj)) #this is v_1 (wall-normal velocity component)
p_2d=np.reshape(p,(ni,nj))   #this is p   (pressure)

#v1_2d=np.transpose(v1_2d)
#v2_2d=np.transpose(v2_2d)
#p_2d=np.transpose(p_2d)

# scale u and v with max u
for i in range (0,ni-1):
   v1_2d[i,:]=v1_2d[i,:]/max(v1_2d[i,:])
   v2_2d[i,:]=v2_2d[i,:]/max(v1_2d[i,:])


blasius=np.genfromtxt("blasius.dat", comments="%")
xi_blas=blasius[:,0]
g_blas=blasius[:,1]
u_blas=blasius[:,2]

#   a control volume, CV. 
#
#  xp(i), yp(j) denote the center of the, CV. u, v and p are stored at (xp,yp)
#
#  xc(i) yc(j) denote the corner (on the high side) of the CV
#
#
#   x-------------------------x  xc(i), yc(j)
#   |                         |
#   |                         |
#   |                         |
#   |                         |
#   |          o xp(i), yp(j) |
#   |                         |
#   |                         |
#   |                         |
#   |                         |
#   x-------------------------x
#
# compute xp
xp=np.zeros(ni)
xp[0]=xc[0]
for i in range (1,ni-1):
   xp[i]=0.5*(xc[i]+xc[i-1])

xp[ni-1]=xc[ni-2]

# compute yp
yp=np.zeros(nj)
yp[0]=yc[0]
for j in range (1,nj-1):
   yp[j]=0.5*(yc[j]+yc[j-1])

yp[nj-1]=yc[nj-2]
#
# make xp and yp 2D arrays
x1_2d=np.zeros((ni,nj))
x2_2d=np.zeros((ni,nj))
for i in range(0,ni-1):
   x2_2d[i,:]=yp

for j in range(0,nj-1):
   x1_2d[:,j]=xp

#
# compute the gradient dudx, dudy
dudx, dudy=np.gradient(v1_2d,xp,yp)


fig_size = (7.5, 7.5)
#************
# velocity profile plot
fig1 = plt.figure("Figure 1")
plt.subplots_adjust(left=0.20,bottom=0.20)
i=170 # plot the velocity profile for i=170
plt.plot(v1_2d[i,:],x2_2d[i,:],'b-')
i=5 # plot the velocity profile for i=5
plt.plot(v1_2d[i,:],x2_2d[i,:],'r--')  #red dashed line
plt.title('Velocity profile')
plt.axis([0,1.1,0,0.05]) # set x & y axis
plt.xlabel('$V_1$') 
plt.ylabel('$x_2$') 
plt.text(0.04,0.04,'$x_1=0.14$ and $1.58$') # show this text at (0.04,0.04)
plt.show()
plt.savefig('velprof.png')

################################ contour plot of v1
fig2 = plt.figure("Figure 2")
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.contourf(x1_2d,x2_2d,v1_2d, 50)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.clim(0.1,1.)
plt.title("contour pressure plot")
plt.axis('scaled')
# only three xticks
plt.locator_params(axis='x', nbins=2)
#ax1.xaxis.set_major_locator(xticks)
plt.axis([0,0.1,0,0.1]) # zoom-in on the first 0.1m from the inlet
plt.title('$V_1$')
plt.show()
plt.savefig('v1_iso.png')

################################ compute the velocity gradient dv_1/dx_2
fig3 = plt.figure("Figure 3")
plt.subplots_adjust(left=0.20,bottom=0.20)
dv1_dx2=np.zeros((ni,nj))
i=170 # plot the velocity gradient for i=170
plt.plot(dudy[i,:],yp)
plt.axis([0,100,0,0.05]) # set x & y axis
plt.title('Velocity gradient')
plt.xlabel(r'$\partial v_1/\partial x_2$')
plt.ylabel('$x_2$') 
plt.text(-380,0.004,'$x_1=0.52$')
plt.show()
plt.savefig('v1_grad.png')

#%%
##################################################
# E1
##################################################
new_prob(1)

# Substask 1
plt.figure()
i=170 # plot the velocity profile for i=170
plt.plot(v1_2d[i,:], x2_2d[i,:], 'b-', label='$x_1=1.58$')

i=85 # plot the velocity profile for i=85
plt.plot(v1_2d[i,:], x2_2d[i,:], 'g-.', label='$x_1=0.66$')

i=5 # plot the velocity profile for i=5
plt.plot(v1_2d[i,:], x2_2d[i,:], 'r--', label='$x_1=0.14$')

plt.title('Velocity profile')
plt.axis([0,1.1,0,0.05])
plt.xlabel('$v_1$') 
plt.ylabel('$x_2$') 
plt.legend(loc='best')
plt.savefig('E1_1', dpi=dpi, bbox_inches='tight')
plt.show()

# Substask 2
'''
evaluate the parenthesis at different values for x1
then plug in the term in the equation

also add more plots to the figures
'''

g_prim_blas = u_blas # Pretty sure that g'' in the blasius.dat file is incorrectly named

plt.figure()

i = 170 # plot the velocity profile for i=170

# Freestream velocity at x1 location
V1_inf = v1_2d[i,-1]

xi = x2_2d[i,:] * np.sqrt(V1_inf / (viscos * x1_2d[i,:])) # eqn (3.48), x2 = yp

v1_norm = v1_2d[i, :] / V1_inf
v2_norm = v2_2d[i, :] / V1_inf

x1_value = x1_2d[i,0]
v2_blas = - (1/2) * np.sqrt(viscos * V1_inf/x1_value) * (g_blas - xi_blas * g_prim_blas)
v2_blas_norm = v2_blas / V1_inf

plt.plot(xi, v1_norm, 'b-', label=fr'$x_1={xc[i]:.2f}$')
plt.plot(xi_blas, u_blas, 'k--', label=fr'Blasius solution, $x_1={xc[i]:.2f}$')

plt.title(fr'Normalized velocities for $v_1$')
plt.axis([0,20,0,1.1])
plt.xlabel(r'$\xi$')
plt.ylabel(r'$v_1 / V_{1,\infty}$')
plt.legend()
plt.grid(True)
plt.savefig('E1_2', dpi=dpi, bbox_inches='tight')
plt.show()


# Plot
plt.figure()

plt.plot(xi, v2_norm, 'r-', label=fr'$x_1={xc[i]:.2f}$')
plt.plot(xi_blas, v2_blas_norm, 'k--', label=fr'Blasius solution, $x_1={xc[i]:.2f}$')

plt.title(fr'Normalized velocities for $v_2$')
plt.axis([0,20,0,0.003])
plt.xlabel(r'$\xi$')
plt.ylabel(r'$v_2 / V_{1,\infty}$')
plt.legend()
plt.grid(True)
plt.savefig('E1_3', dpi=dpi, bbox_inches='tight')
plt.show()


# i = 50 # plot the velocity profile for i=170

# # Freestream velocity at x1 location
# V1_inf = v1_2d[i,-1]

# xi = x2_2d[i,:] * np.sqrt(V1_inf / (viscos * x1_2d[i,:])) # eqn (3.48), x2 = yp

# v1_norm = v1_2d[i, :] / V1_inf
# v2_norm = v2_2d[i, :] / V1_inf

# x1_value = x1_2d[i,0]
# v2_blas = - (1/2) * np.sqrt(viscos * V1_inf/x1_value) * (g_blas - xi_blas * g_prim_blas)
# v2_blas_norm = v2_blas / V1_inf

# plt.plot(xi, v1_norm, 'b-', label=fr'$x_1={xc[i]:.2f}$')
# plt.plot(xi_blas, u_blas, 'k--', label=fr'Blasius solution, $x_1={xc[i]:.2f}$')

# plt.title(fr'Normalized velocities for $v_1$')
# plt.axis([0,20,0,1.1])
# plt.xlabel(r'$\xi$')
# plt.ylabel(r'$v_1 / V_{1,\infty}$')
# plt.legend()
# plt.grid(True)
# plt.savefig('E1_2', dpi=dpi, bbox_inches='tight')
# plt.show()


# Plot
plt.figure()

plt.plot(xi, v2_norm, 'r-', label=fr'$x_1={xc[i]:.2f}$')
plt.plot(xi_blas, v2_blas_norm, 'k--', label=fr'Blasius solution, $x_1={xc[i]:.2f}$')

plt.title(fr'Normalized velocities for $v_2$')
plt.axis([0,20,0,0.003])
plt.xlabel(r'$\xi$')
plt.ylabel(r'$v_2 / V_{1,\infty}$')
plt.legend()
plt.grid(True)
plt.savefig('E1_3', dpi=dpi, bbox_inches='tight')
plt.show()


#%%
##################################################
# E2
##################################################
new_prob(2)

# Need to now plot delta_gg and delta_99_blasius as a function of x1
delta_99_blasius = 5 * np.sqrt((viscos * xc)/V1_inf)
delta_99 = np.zeros_like(xc)

for i in range(len(xc)):
    V1_inf = v1_2d[i,-1]
    index_99 = np.where(v1_2d[i,:] >= 0.99 * V1_inf)[0][0]
    delta_99[i] = x2_2d[i,index_99]

plt.figure()
plt.plot(xc, delta_99_blasius, 'b-', label=r'$\delta_{99, \text{Blasius}}$')
plt.plot(xc, delta_99, 'r-', label=r'$\delta_{99}$')
plt.xlabel(r'$x_1$')
plt.ylabel(fr'Boundary layer thickness')
plt.title('Boundary layer thickness')
plt.legend()
plt.grid(True)
plt.savefig('E2_1', dpi=dpi, bbox_inches='tight')
plt.show()


# Creating empty arrays to store results
delta_star = np.zeros(ni)
theta = np.zeros(ni)

delta_star_blasius = np.zeros(ni)
theta_blasius = np.zeros(ni)

for i in range(ni):
     
    x2 = x2_2d[i, :] # local x2 array
    u = v1_2d[i, :].copy() # local velocity profile

    # local freestream velocity (could also use constant V1_inf = 1.0)
    V1_inf = np.max(v1_2d[i,:])
    if V1_inf == 0:
        # avoid division by zero, skipping or setting to NaN
        delta_star[i] = np.nan
        theta[i] = np.nan
        continue
    
    uhat = v1_2d[i,:] / V1_inf # normalized velocity
    jmax = np.argmax(u)  # finding index of maximum u

    # integration domain up to jmax (include jmax)
    x2_int = x2[:jmax+1]
    uhat_int = uhat[:jmax+1]

    # integrands
    integrand_delta = 1.0 - uhat_int
    integrand_theta = uhat_int * (1.0 - uhat_int)

    # trapezoidal integration
    delta_star[i] = np.trapezoid(integrand_delta, x2_int)
    theta[i]      = np.trapezoid(integrand_theta, x2_int)

    delta_star_blasius[i] = 1.721 * np.sqrt((viscos * xc[i-1]) / V1_inf)
    theta_blasius[i]      = 0.664 * np.sqrt((viscos * xc[i-1]) / V1_inf)
    
# Plotting
x1 = x1_2d[:,0]
plt.figure()
plt.plot(x1, delta_star, label="δ* (numerical)")
plt.plot(x1, theta, label=r'$\theta$ (numerical)')
plt.plot(x1, delta_star_blasius, 'k--', label="δ* (Blasius)")
plt.plot(x1, theta_blasius, 'r--', label=r'$\theta$ (Blasius)')
plt.xlabel('$x_1$')
plt.ylabel('Boundary layer thickness')
plt.legend()
plt.grid(True)
plt.savefig('E2_2', dpi=dpi, bbox_inches='tight')
plt.show()

#display(delta_star, theta)

# %%
##################################################
# E3
##################################################
new_prob(3)

plt.figure()
i = 85
gradient = np.gradient(v1_2d[i,:], x2_2d[i,:])
plt.plot(gradient, x2_2d[i,:], 'r--', label=fr'$x_1={xc[i]:.2f}$')
i = 170
gradient = np.gradient(v1_2d[i,:], x2_2d[i,:])
plt.plot(gradient, x2_2d[i,:], 'b--', label=fr'$x_1={xc[i]:.2f}$')
i = 100
gradient = np.gradient(v1_2d[i,:], x2_2d[i,:])
plt.plot(gradient, x2_2d[i,:], 'g--', label=fr'$x_1={xc[i]:.2f}$')

plt.title('Velocity gradients')
plt.axis([-5,100,0,0.04])
plt.legend()
plt.ylabel('$x_2$')
plt.xlabel('$\\partial V_1 / \\partial x_2$')
plt.savefig('E3_1', dpi=dpi, bbox_inches='tight')
plt.show()

print('prove/disprove assumptions')

# %%
##################################################
# E4
##################################################
new_prob(4)

V1_inf = 1 # freestream velocity (assumed constant according to problem statement)
rho = 1.204             # kg/m^3 at 20C
viscos_dyn = viscos / rho

# compute the gradient dudx, dudy
dvdx, dvdy=np.gradient(v2_2d,xp,yp)

# dvdx = np.gradient(v2_2d, xp, axis=0)  # ∂v2/∂x1

tau_12 = viscos_dyn * (dudy + dvdx)   # shape (ni, nj) as denoted in boundary_layer.py

# extract wall shear stress at the wall index j=0
tau_w = tau_12[:,0]          

# local skin friction coefficient
Cf = tau_w / (0.5 * rho * V1_inf**2)

# Blasius solution for Cf
Cf_blas = 0.664 / np.sqrt((V1_inf * x1) / viscos)

# Plotting results
valid = np.isfinite(Cf) & (np.sqrt((V1_inf * x1) / viscos) > 0) # Avoiding invalid values

plt.figure()
plt.plot(xp[valid], Cf[valid], label='$C_f$')
plt.plot(xp[valid], Cf_blas[valid], '--', label='Blasius solution')

plt.title(fr'Skinfriction at different $x_1$ locations')
plt.xlabel('$x_1$')
plt.ylabel('$C_f$')
plt.legend()
plt.grid(True)
plt.savefig('E4_1', dpi=dpi, bbox_inches='tight')
plt.show()

# %%
##################################################
# E5
##################################################
new_prob(5)

omega_3 = dvdx - dudy

# print('dvdx: ', dvdx)
# print('dudy: ', dudy)
# print('vorticity: ',omega_3)

plt.figure()
i = 50
plt.plot(omega_3[i,:], x2_2d[i,:], label=fr'$x_1={xc[i]:.2f}$')
i=100
plt.plot(omega_3[i,:], x2_2d[i,:], '-', label=fr'$x_1={xc[i]:.2f}$')
i=170
plt.plot(omega_3[i,:], x2_2d[i,:], '-', label=fr'$x_1={xc[i]:.2f}$')

plt.title(fr'Vorticity at different $x_1$ locations (above plate)')
plt.axis([-160,1,-0,0.05])
plt.xlabel('$\omega$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.savefig('E5_1', dpi=dpi, bbox_inches='tight')
plt.show()

plt.figure()
i=0
plt.plot(omega_3[i,2:], x2_2d[i,2:], '-', label=fr'$x_1={xc[i]:.2f}$')
i=19
plt.plot(omega_3[i,:], x2_2d[i,:], '-', label=fr'$x_1={xc[i]:.2f}$')
i=20
plt.plot(omega_3[i,:], x2_2d[i,:], '-', label=fr'$x_1={xc[i]:.2f}$')

plt.title(fr'Vorticity at different $x_1$ locations (upstream of plate)')
plt.axis([-2000,200,-0,0.005])
plt.xlabel('$\omega$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.savefig('E5_2', dpi=dpi, bbox_inches='tight')
plt.show()

print(omega_3[i,:])
# %%
##################################################
# E6
##################################################
new_prob(6)

# S12 = 0.5 * (dudy + dvdx)
# Omega12 = 0.5 * (dudy - dvdx)

# exclude value at corner of the domain
dudx_cut, dudy_cut=np.gradient(v1_2d[1:,:],xp[1:],yp[:])
dvdx_cut, dvdy_cut=np.gradient(v2_2d[1:,:],xp[1:],yp[:])

S12 = 0.5 * (dudy + dvdx)
Omega12 = 0.5 * (dudy - dvdx)

S12_cut = 0.5 * (dudy_cut + dvdx_cut)
Omega12_cut = 0.5 * (dudy_cut - dvdx_cut)

print('dudy: ', dudy)
print('dvdx: ', dvdx) # basically = 0
print('max value of dvdx: ', np.unravel_index(np.argmax(dvdx, axis=None), dvdx.shape)[0])
# print(S12)
print(Omega12[20,0])
print(dudy[20,0])

max_i = np.unravel_index(np.argmax(S12_cut, axis=None), S12_cut.shape)[0]

print('i =', max_i, 'Max S12 at i:', np.max(S12_cut))

max_i = np.unravel_index(np.argmax(Omega12_cut, axis=None), Omega12_cut.shape)[0]

print('i =', max_i, 'Max omega12 at i:', np.max(Omega12_cut))

# i = 170

# plt.figure()
# plt.plot(S12[i,:], yp, 'r-', label='$S_{12}$')
# plt.plot(Omega12[i,:], yp, 'b--', label='$\Omega_{12}$')
# plt.title(fr'Shear and Vorticity at $x_1={xc[i]:.2f}$')
# plt.xlabel('$S_{12}, \\Omega_{12}$')
# plt.ylabel('$x_2$')
# plt.legend()
# plt.axis([-0.5,35,0,0.05])
# plt.grid(True)
# plt.savefig('E6_1', dpi=dpi, bbox_inches='tight')
# plt.show()

# i = 85

# plt.figure()
# plt.plot(S12[i,:], yp, 'r-', label='$S_{12}$')
# plt.plot(Omega12[i,:], yp, 'b--', label='$\Omega_{12}$')
# plt.title(fr'Shear and Vorticity at $x_1={xc[i]:.2f}$')
# plt.xlabel('$S_{12}, \\Omega_{12}$')
# plt.ylabel('$x_2$')
# plt.legend()
# plt.grid(True)
# plt.axis([-0.5,55,0,0.05])
# plt.savefig('E6_2', dpi=dpi, bbox_inches='tight')
# plt.show()

# i = 5

# plt.figure()
# plt.plot(S12[i,:], yp, 'r-', label='$S_{12}$')
# plt.plot(Omega12[i,:], yp, 'b--', label='$\Omega_{12}$')
# plt.title(fr'Shear and Vorticity at $x_1={xc[i]:.2f}$')
# plt.xlabel('$S_{12}, \\Omega_{12}$')
# plt.ylabel('$x_2$')
# plt.legend()
# plt.grid(True)
# plt.savefig('E6_3', dpi=dpi, bbox_inches='tight')
# plt.show()


plt.figure()

i = 170-1
plt.plot(S12[i,:], yp, 'r-', label='$S_{12}$' + fr', $x_1={xc[i]:.2f}$')
plt.plot(Omega12[i,:], yp, 'b--', label='$\Omega_{12}$' + fr', $x_1={xc[i]:.2f}$')

i = 50
plt.plot(S12[i,:], yp, 'g-', label='$S_{12}$' + fr', $x_1={xc[i]:.2f}$')
plt.plot(Omega12[i,:], yp, 'y--', label='$\Omega_{12}$' + fr', $x_1={xc[i]:.2f}$')

i = 1
plt.plot(S12[i,:], yp, 'm-', label='$S_{12}$' + fr', $x_1={xc[i]:.2f}$')
plt.plot(Omega12[i,:], yp, 'k--', label='$\Omega_{12}$' + fr', $x_1={xc[i]:.2f}$')

plt.title(fr'Shear and Vorticity')
plt.axis([-1,80,0,0.04])
plt.xlabel('$S_{12}, \\Omega_{12}$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.savefig('E6_4', dpi=dpi, bbox_inches='tight')
plt.show()


plt.figure()

i = 20
plt.plot(S12[i,:], yp, 'r-', label='$S_{12}$' + fr', $x_1={xc[i]:.2f}$')
plt.plot(Omega12[i,:], yp, 'b--', label='$\Omega_{12}$' + fr', $x_1={xc[i]:.2f}$')

i = 19
plt.plot(S12[i,:], yp, 'g-', label='$S_{12}$' + fr', $x_1={xc[i]:.2f}$')
plt.plot(Omega12[i,:], yp, 'y--', label='$\Omega_{12}$' + fr', $x_1={xc[i]:.2f}$')

i = 18
plt.plot(S12[i,:], yp, 'm-', label='$S_{12}$' + fr', $x_1={xc[i]:.2f}$')
plt.plot(Omega12[i,:], yp, 'k--', label='$\Omega_{12}$' + fr', $x_1={xc[i]:.2f}$')

plt.title(fr'Shear and Vorticity, plate border')
plt.axis([-100,1000,0,0.005])
plt.xlabel('$S_{12}, \\Omega_{12}$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.savefig('E6_5', dpi=dpi, bbox_inches='tight')
plt.show()

# %%
##################################################
# E7
##################################################
new_prob(7)

# Phi = viscos * np.array([[2 *dudx * dudx, (dudy + dvdx) * dudy],
#                          [(dvdx + dudy) * dvdx, 2 *dvdy * dvdy]])

S_ij =  np.array([[dudx, 0.5 * (dudy + dvdx)],
                [0.5 * (dvdx + dudy), dvdy]])

grad_v_ij = np.array([[dudx, dudy],
                      [dvdx, dvdy]])

print(np.shape(S_ij), np.shape(grad_v_ij))

tau_ij = 2 * viscos * S_ij

Phi = np.einsum('jikl, ijkl -> kl',tau_ij, grad_v_ij)
# Phi = tau_ij @ grad_v_ij
# Phi = tau_ij * grad_v_ij
# Phi = np.dot(tau_ij, grad_v_ij)
# Phi = np.sum(tau_ij.T * grad_v_ij, axis=(0, 1))
# Phi = tau_ij.T @ grad_v_ij
# Phi = 2 * viscos * np.sum(S_ij * grad_v_ij, axis=(0, 1))

print(Phi)
print('shape of Phi: ', np.shape(Phi))
print("Shape of dudx:", np.shape(dudx))
print("Shape of S_ij:", np.shape(S_ij))
print("Shape of grad_v_ij:", np.shape(grad_v_ij))
print("Shape of tau_ij:", np.shape(tau_ij))
print("Shape of Phi:", np.shape(Phi))

print("Contains NaN:", np.any(np.isnan(Phi)))
print("Contains inf:", np.any(np.isinf(Phi)))
print("All zeros:", np.all(Phi == 0))

print("Phi stats:")
print("  Min:", np.min(Phi))
print("  Max:", np.max(Phi))
print("  Mean:", np.mean(Phi))
print("  Std:", np.std(Phi))

print("  Order of magnitude:", np.log10(np.abs(Phi[Phi != 0]).mean()))

fig2 = plt.figure()
# plt.contourf(x1_2d[50,:], x2_2d[50,:], Phi, 50)
# plt.pcolormesh(x1_2d, x2_2d, Phi, shading = 'auto', cmap='plasma')
# plt.plot(x1_2d[50,:], x2_2d[50,:], Phi)
# plt.plot(x1_2d[50,:], x2_2d[50,:], Phi[50,:])
Phi_log = np.log10(Phi)

plt.contourf(x1_2d, x2_2d, Phi_log, levels=50, cmap='viridis')
# plt.contour(x1_2d, x2_2d, Phi_plot)
# plt.pcolormesh(x1_2d, x2_2d, Phi, shading = 'auto', cmap='plasma')


plt.axis([-0.2,xc[-1],0, yc[-1]])
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("contour dissipation ($\Phi_1$)")
plt.colorbar(label=r'$\log_{10}(\Phi)$')
plt.savefig('E7_1', dpi=dpi, bbox_inches='tight')
plt.show()


fig2 = plt.figure()

plt.contourf(x1_2d, x2_2d, Phi_log, levels=20, cmap='viridis')

plt.axis([-0.05,0.15,0, 0.003])
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("contour dissipation ($\Phi_1$), plate border")
plt.colorbar(label=r'$\log_{10}(\Phi)$')
plt.savefig('E7_2', dpi=dpi, bbox_inches='tight')
plt.show()

from scipy import integrate

# Double integral over the entire domain
integral = integrate.trapezoid(integrate.trapezoid(Phi, xp, axis=0), yp)
print(f"Double integral of Phi: {integral:.20f}")

rho = 1.204   # air density [kg/m^3]
cp  = 1005    # specific heat at 20C [J/kg/K]

denominator = 1 / (-2 * rho * cp) * integral
v1_int = v1_2d[0,:]

numerator = np.trapezoid(v1_int, x2)

T_b = denominator / numerator

print('bulk temperature: ', T_b)

# %%
##################################################
# E8
##################################################
new_prob(8)

tau_11 = viscos * (dudx + dudx)
tau_12 = tau_12
tau_21 = tau_12 # symmetric tensor
tau_22 = viscos * (dvdy + dvdy)

ni, nj = tau_11.shape
eigvals = np.zeros((ni, nj, 2)) # Creates a vector of length 2 for each (i,j) point
eigvecs = np.zeros((ni, nj, 2, 2)) # Creates a 2x2 matrix for each (i,j) point

for i in range(ni): # Loop over all ni values
    for j in range(nj): # Loop over all nj values
        tau = np.array([[tau_11[i,j], tau_12[i,j]],
                        [tau_21[i,j], tau_22[i,j]]])
        vals, vecs = np.linalg.eig(tau)
        eigvals[i,j] = vals
        eigvecs[i,j] = vecs

#print('Eigenvalues', eigvals)
#print('Eigenvectors', eigvecs)

i = 200  
eigval_slice = eigvals[i, :, 1]   
tau11_slice = tau_11[i, :]
tau12_slice = tau_12[i, :]
tau21_slice = tau_21[i, :]
tau22_slice = tau_22[i, :]


# Eigenvalue vs vertical position
plt.figure()

plt.plot(eigval_slice, yp, label="Eigenvalue")

plt.title(fr"Eigenvalue vs $x_2$ at $x_1$[i = {i}]")
plt.ylabel(fr"$x_2$")
plt.xlabel("$\lambda$")
plt.axis([-0.0001,0.001,0,0.1])
plt.legend()
plt.savefig('E8_1', dpi=dpi, bbox_inches='tight')
plt.show()

plt.figure()

plt.plot(tau12_slice, yp, label=r"$\tau_{12} = \tau_{21}$")
plt.plot(tau11_slice, yp, label=r"$\tau_{11}$")
# plt.plot(tau21_slice, yp, label=r"$\tau_{21}$")
plt.plot(tau22_slice, yp, label=r"$\tau_{22}$")

plt.title(fr"Stress components vs $x_2$ at $x_1$[i = {i}]")
plt.axis([-5e-6,5e-6,0,0.25])
# plt.axis([0,0.3,-0.000005,0.000008])
plt.ylabel(fr"$x_2$")
plt.xlabel("Stress component value")
plt.legend()
plt.savefig('E8_2', dpi=dpi, bbox_inches='tight')
plt.show()


plt.figure()

plt.plot(tau12_slice, yp, label=r"$\tau_{12} = \tau_{21}$")

plt.title(fr"Stress components vs $x_2$ at $x_1$[i = {i}]")
plt.axis([-0.0001,0.001,0,0.1])
plt.ylabel(fr"$x_2$")
plt.xlabel("Stress component value")
plt.legend()
plt.savefig('E8_3', dpi=dpi, bbox_inches='tight')
plt.show()


print(eigvecs[1,4])

# %%
##################################################
# E9
##################################################
new_prob(9)

principal_vecs = eigvecs[:,:, :,0]   # Picking the first row of the eigenvector matrix
# Each eigenvalue has two eigenvectors

# extract components
u_quiver = principal_vecs[:,:,0]   # x-component of eigenvector
v_quiver = principal_vecs[:,:,1]   # y-component of eigenvector

plt.figure()
plt.quiver(x1_2d, x2_2d, u_quiver, v_quiver, 
           scale=20, width=0.002, color="blue")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Principal stress eigenvector field all points")
plt.axis("scaled")
plt.savefig('E9_1', dpi=dpi, bbox_inches='tight')
plt.show()

step = 5  # plotting every 5th point
plt.figure()
plt.quiver(x1_2d[::step, ::step], x2_2d[::step, ::step],
           u_quiver[::step, ::step], v_quiver[::step, ::step],
           scale=70, width=0.002, color="blue")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Principal stress eigenvector field every 5th point")
plt.axis("scaled")
plt.savefig('E9_2', dpi=dpi, bbox_inches='tight')
plt.show()

step = 10  # plotting every 5th point
plt.figure()
plt.quiver(x1_2d[::step, ::step], x2_2d[::step, ::step],
           u_quiver[::step, ::step], v_quiver[::step, ::step],
           scale=30, width=0.002, color="blue")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Principal stress eigenvector field every 10th point")
plt.axis("scaled")
plt.savefig('E9_3', dpi=dpi, bbox_inches='tight')
plt.show()



# %%
##################################################
# E10
##################################################
new_prob(10)

vorticity = (dvdx - dudy)

d2v1_dx2 = np.gradient(dudy, yp, axis=1)

term1 = v1_2d * dudx
term2 = v2_2d * dudy
term3 = viscos * d2v1_dx2

left_side = term1 + term2
right_side = term3

residual = left_side - right_side

plt.figure()

i = 5
plt.plot(term1[i,:], yp, 'r-', label=fr'$v_1 \cdot \partial v_1/ \partial x_1$')
plt.plot(term2[i,:], yp, 'g-', label=fr'$v_2 \cdot \partial v_1/ \partial x_2$')
plt.plot(term3[i,:], yp, 'b--', label=fr'$\nu \cdot \partial^2 v_1/ \partial x_2^2$')


plt.title(fr'$x_1$[i = {i}] $\approx${xp[i]:.3f}')
# plt.axis([0,0.03,-0.4,0.2])
plt.ylabel('$x_2$')
plt.xlabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('E10_1', dpi=dpi, bbox_inches='tight')
plt.show()

plt.figure()

i = 5
plt.plot(left_side[i,:], yp, 'r-', label='LHS (sum)')
plt.plot(right_side[i,:], yp, 'g', label='RHS')
plt.plot(residual[i,:], yp, 'b--', label='Residual')


plt.title(fr'$x_1$[i = {i}] $\approx${xp[i]:.3f}')
# plt.axis([0,0.03,-0.4,0.2])
plt.ylabel('$x_2$')
plt.xlabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('E10_2', dpi=dpi, bbox_inches='tight')
plt.show()



plt.figure()

i = 85
plt.plot(term1[i,:], yp, label=fr'$v_1 \cdot \partial v_1/ \partial x_1$')
plt.plot(term2[i,:], yp, label=fr'$v_2 \cdot \partial v_1/ \partial x_2$')
plt.plot(term3[i,:], yp, label=fr'$\nu \cdot \partial^2 v_1/ \partial x_2^2$')
plt.plot(left_side[i,:], yp, '--', label='LHS (sum)')
plt.plot(right_side[i,:], yp, ':', label='RHS')
plt.plot(residual[i,:], yp, '-.', label='Residual')

plt.title(fr'$x_1$[i = {i}] $\approx${xp[i]:.3f}')
plt.axis([-0.4,0.2,0,0.03])
plt.ylabel('$x_2$')
plt.xlabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('E10_3', dpi=dpi, bbox_inches='tight')
plt.show()



plt.figure()

i = 170
plt.plot(term1[i,:], yp, label=fr'$v_1 \cdot \partial v_1/ \partial x_1$')
plt.plot(term2[i,:], yp, label=fr'$v_2 \cdot \partial v_1/ \partial x_2$')
plt.plot(term3[i,:], yp, label=fr'$\nu \cdot \partial^2 v_1/ \partial x_2^2$')
plt.plot(left_side[i,:], yp, '--', label='LHS (sum)')
plt.plot(right_side[i,:], yp, ':', label='RHS')
plt.plot(residual[i,:], yp, '-.', label='Residual')

plt.title(fr'$x_1$[i = {i}] $\approx${xp[i]:.3f}')
plt.axis([-0.4,0.2,0,0.03])
plt.ylabel('$x_2$')
plt.xlabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('E10_4', dpi=dpi, bbox_inches='tight')
plt.show()

dpdx1, dpdx2 = np.gradient(p_2d, xp, yp, edge_order=2)

plt.figure()

i = 25
plt.plot(dpdx1[i, :], yp, label=fr"$x_1=${xc[i]:.2f}")

i = 50
plt.plot(dpdx1[i, :], yp, label=f"x1={xc[i]:.2f}")

i = 100
plt.plot(dpdx1[i, :], yp, label=f"x1={xc[i]:.2f}")

i = 150
plt.plot(dpdx1[i, :], yp, label=f"x1={xc[i]:.2f}")

i = 200
plt.plot(dpdx1[i, :], yp, label=f"x1={xc[i]:.2f}")

plt.title(fr"Pressure gradient at $x_1$")
plt.axis([-0.075,0.01, -0.01,0.8])
plt.ylabel("$x_2$")
plt.xlabel(r"$\partial p / \partial x_1$")
plt.grid(True)
plt.legend()
plt.savefig('E10_5', dpi=dpi, bbox_inches='tight')
plt.show()



# 2) second derivative ∂^2 v1 / ∂ x2^2
d2v1_dx1dx2, d2v1_dx2sq = np.gradient(dudy, xp, yp, edge_order=2)

d2_at_wall = d2v1_dx2sq[:, 0] # Extracting values at the wall (j=0)

d2_at_wall_log = np.log(d2_at_wall)

plt.figure()

plt.plot(xp, d2_at_wall, '-o', markersize=3)

plt.title('Second derivative of $v_1$ at the wall')
# plt.axis([-0.5, 2.5,-10e10,10e15])
plt.axis([-0.2,0.5,-0.0006*10**8,0.0006*10**8])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$\partial^2 v_1/\partial x_2^2\ \mathrm{at}\ x_2=0$')
# plt.yscale('log')
plt.grid(True)
plt.savefig('E10_6', dpi=dpi, bbox_inches='tight')
plt.show()

plt.figure()

plt.plot(xp, d2_at_wall_log, '-o', markersize=3)

plt.title('Second derivative of $v_1$ at the wall, log scale')
plt.axis([-0.2, 0.5,-1,15])
# plt.axis([-0.2,0.5,-0.0006*10**8,0.0006*10**8])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$\partial^2 v_1/\partial x_2^2\ \mathrm{at}\ x_2=0$')
# plt.yscale('log')
plt.grid(True)
plt.savefig('E10_7', dpi=dpi, bbox_inches='tight')
plt.show()

max_i = np.unravel_index(np.argmax(d2_at_wall, axis=None), d2_at_wall.shape)[0]

print('i =', max_i, 'Max dissipation at i =', np.max(d2_at_wall))


i = 50
print(f"x1 ≈ {xp[i]:.4f}, d2v1/dx2^2 at wall = {d2v1_dx2sq[i,0]:.6e}")


dvorticity_dx1, dvorticity_dx2 = np.gradient(vorticity, xp, yp, edge_order=2)

dvorticity_dx2_wall = dvorticity_dx2[:, 0]
dvorticity_dx2_wall_log = np.log(dvorticity_dx2_wall)

plt.figure()

plt.plot(xp, dvorticity_dx2_wall, '-o', markersize=3)

plt.title("Gradient of vorticity at the wall")
# plt.axis([-0.5,2.5,-1,15])
plt.axis([-0.2,0.5,-0.0006*10**8,0.0006*10**8])
plt.xlabel("$x_1$")
plt.ylabel(r"$\partial \omega / \partial x_2$ at wall")
# plt.yscale('log')
plt.grid(True)
plt.savefig('E10_8', dpi=dpi, bbox_inches='tight')
plt.show()


plt.figure()

plt.plot(xp, dvorticity_dx2_wall_log, '-o', markersize=3)

plt.title("Gradient of vorticity at the wall, log scale")
plt.axis([-0.2,0.5,-1,15])
# plt.axis([-0.2,0.5,-0.0006*10**8,0.0006*10**8])
plt.xlabel("$x_1$")
plt.ylabel(r"$\partial \omega / \partial x_2$ at wall")
# plt.yscale('log')
plt.grid(True)
plt.savefig('E10_9', dpi=dpi, bbox_inches='tight')
plt.show()


print(dvorticity_dx2_wall + d2_at_wall)
# %%