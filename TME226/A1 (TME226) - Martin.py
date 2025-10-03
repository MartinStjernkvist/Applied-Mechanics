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
plt.rc('figure', figsize=(5, 4))


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



# plotta_subplots(
#     nrows=1,
#     ncols=2,
#     fig_size=(12, 5),
#     fig_title="MultiPlot",
#     subplot_instructions=[
#         {
#             "plotstyle": "plot",
#             "x1_data": v1_2d[100, :],
#             "x2_data": x2_2d[100, :],
#             "color": "b",
#             "title": "Line Plot",
#             "xlabel": "v1",
#             "ylabel": "x2"
#         },
#         {
#             "plotstyle": "contourf",
#             "x1_data": x1_2d,
#             "x2_data": x2_2d,
#             "x3_data": v1_2d,
#             "cmap": "plasma",
#             "title": "Contour Plot",
#             "xlabel": "x1",
#             "ylabel": "x2"
#         }
#     ],
#     suptitle="Velocity Profiles"
# )

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


'''
def nu(xi, x1, x2, V1_inf):
    return (x2 / xi)**2 * (V1_inf / x1)

# eqn (3.49)
def xi(x1, x2, nu, V1_inf):
    return x2 * (V1_inf / (nu * x1))**(1/2)



print('Velocity (v1) profiles at different values for x1, plotted against x2:')

frame = [0,1,0,0.05]

plotta_subplots(
    nrows=1,
    ncols=3,
    fig_title="v2 vs x2",
    subplot_instructions=[
        {
            "plotstyle": "plot",
            "x1_data": v1_2d[50, :],
            "x2_data": x2_2d[50, :],
            "color": "b",
            "title": "i=50",
            "xlabel": "v1",
            "ylabel": "x2",
            'frame': frame
        },
        {
            "plotstyle": "plot",
            "x1_data": v1_2d[100, :],
            "x2_data": x2_2d[100, :],
            "color": "b",
            "title": "i=100",
            "xlabel": "v1",
            "ylabel": "x2",
            'frame': frame
        },
        {
            "plotstyle": "plot",
            "x1_data": v1_2d[150, :],
            "x2_data": x2_2d[150, :],
            "color": "b",
            "title": "i=150",
            "xlabel": "v1",
            "ylabel": "x2",
            'frame': frame
        }
    ],
    suptitle="Velocity Profiles (v1 vs x2)"
)
'''

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
plt.xlabel('$V_1$') 
plt.ylabel('$x_2$') 
plt.legend(loc='best')
plt.savefig('velprof.png')
plt.show()


# Substask 2
i = 170 # plot the dimensioneless velocity profile for i=170

# Freestream velocity at x1 location
V1_inf = v1_2d[i,-1] # Double check this value
# print('V1_inf:', V1_inf)

xi = x2_2d[i,:] * np.sqrt(V1_inf / (viscos * x1_2d[i,:])) # eqn (3.48), x2 = yp
# print(xi)

v1_norm = v1_2d[i, :] / V1_inf
v2_norm = v2_2d[i, :] / V1_inf
#print(v2_norm)

g_prim_blas = u_blas # Pretty sure that g'' in the blasius.dat file is incorrectly named
'''
evaluate the parenthesis at different values for x1
then plug in the term in the equation

also add more plots to the figures
'''

x1_value = 1.58
value = np.sqrt(viscos * V1_inf/x1_value)
v2_blas = - (1/2) * value * (g_blas - xi_blas * g_prim_blas) # combine eqn (3.44), eqn (3.48) & eqn (3.50)
v2_blas_norm = v2_blas / V1_inf

# Plot
plt.figure()
plt.plot(xi, v1_norm, 'b-', label=fr'$x_1={xc[i]:.2f}$')

# Add Blasius solution for comparison
plt.plot(xi_blas, u_blas, 'k--', label='Blasius solution, v1_Blasius_norm')

plt.title('Similarity velocity profile')
plt.axis([0,20,0,1.1])
plt.xlabel(r'$\xi$')
plt.ylabel(r'$v_1 / V_{1,\infty}$')
plt.legend()
plt.grid(True)
plt.show()

# Plot
plt.figure()
plt.plot(xi, v2_norm, 'r--', label=fr'$x_1={xc[i]:.2f}$')

# Add Blasius solution for comparison
plt.plot(xi_blas, v2_blas_norm, 'k--', label='Blasius solution v2_Blasius_norm')

plt.title('Similarity velocity profile')
plt.axis([0,20,0,0.003])
plt.xlabel(r'$\xi$')
plt.ylabel(r'$v_2 / V_{1,\infty}$')
plt.legend()
plt.grid(True)
plt.show()


#%%
##################################################
# E2
##################################################
new_prob(2)

# Need to now plot delta_gg and delta_99_blasius as a function of x1
delta_99_blasius = 5 * np.sqrt((viscos * xc)/V1_inf)
delta_99_all = np.zeros_like(xc)

for i in range(len(xc)):
    V1_inf = v1_2d[i,-1]
    index_99 = np.where(v1_2d[i,:] >= 0.99 * V1_inf)[0][0]
    delta_99_all[i] = x2_2d[i,index_99]

plt.figure()
plt.plot(xc, delta_99_blasius, 'b-', label=r'$\delta_{gg}$')
plt.plot(xc, delta_99_all, 'r--', label=r'$\delta_{99}$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'Boundary layer thickness')
plt.title('Boundary layer thickness vs x1')
plt.legend()
plt.grid(True)
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
plt.ylabel('thickness (m)')
plt.legend()
plt.grid(True)
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
plt.plot(xp[valid], Cf[valid], label='Cf (numerical)')
plt.plot(xp[valid], Cf_blas[valid], '--', label='Cf (Blasius)')
plt.xlabel('$x_1$')
plt.ylabel('$C_f$')
plt.legend()
plt.grid(True)
plt.show()

# %%
##################################################
# E5
##################################################
new_prob(5)

omega_3 = dvdx - dudy

print('dvdx: ', dvdx)
print('dudy: ', dudy)
print('vorticity: ',omega_3)

plt.figure()
i = 85
plt.plot(omega_3[i,:], x2_2d[i,:], label=fr'$x_1={xc[i]:.2f}$')
i=100
plt.plot(omega_3[i,:], x2_2d[i,:], '--', label=fr'$x_1={xc[i]:.2f}$')
i=150
plt.plot(omega_3[i,:], x2_2d[i,:], '--', label=fr'$x_1={xc[i]:.2f}$')
i=0
plt.plot(omega_3[i,:], x2_2d[i,:], '--', label=fr'$x_1={xc[i]:.2f}$')

plt.axis([-110,1,-0,0.05])
plt.xlabel('$\omega$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.show()

# %%
##################################################
# E6
##################################################
new_prob(6)

S12 = 0.5 * (dudy + dvdx)
Omega12 = 0.5 * (dudy - dvdx)

print(S12)
print(Omega12)

i = 170

plt.figure()
plt.plot(xp, S12[:,i], 'r-', label='$S_12$')
plt.plot(xp, Omega12[:,i], 'b-', label=fr'$\Omega_{12}$')
plt.title('Shear and Vorticity at x2=1.58')
plt.xlabel('$x_1$')
plt.ylabel('$S_{12}, \\Omega_{12}$')
plt.legend()
plt.grid(True)
plt.show()

i = 85

plt.figure()
plt.plot(xp, S12[:,i], 'r-', label='$S_12$')
plt.plot(xp, Omega12[:,i], 'b-', label=fr'$\Omega_{12}$')
plt.title('Shear and Vorticity at x2=0.66')
plt.xlabel('$x_1$')
plt.ylabel('$S_{12}, \\Omega_{12}$')
plt.legend()
plt.grid(True)
plt.show()

i = 50

plt.figure()
plt.plot(xp, S12[:,i], 'r-', label='$S_12$')
plt.plot(xp, Omega12[:,i], 'b-', label=fr'$\Omega_{12}$')
plt.title('Shear and Vorticity at x2=0.66')
plt.xlabel('$x_1$')
plt.ylabel('$S_{12}, \\Omega_{12}$')
plt.legend()
plt.grid(True)
plt.show()

# X2 (y) value is incorrect, need to be calculated based on given height and amount of elements.

# Change axels, meaning x on y axis, and then plot for x1 coordinates, meaning x1=i for example.


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

# Check if values are extremely small/large
print("  Order of magnitude:", np.log10(np.abs(Phi[Phi != 0]).mean()))

fig2 = plt.figure("Phi")
# plt.contourf(x1_2d[50,:], x2_2d[50,:], Phi, 50)
# plt.pcolormesh(x1_2d, x2_2d, Phi, shading = 'auto', cmap='plasma')
# plt.plot(x1_2d[50,:], x2_2d[50,:], Phi)
# plt.plot(x1_2d[50,:], x2_2d[50,:], Phi[50,:])
Phi_plot = np.log10(Phi)

plt.contourf(x1_2d, x2_2d, Phi_plot, levels=20, cmap='viridis')
# plt.contour(x1_2d, x2_2d, Phi_plot)
# plt.pcolormesh(x1_2d, x2_2d, Phi, shading = 'auto', cmap='plasma')
plt.colorbar(label=r'$\log_{10}(\Phi)$')
plt.axis([-0.2,xc[-1],0, yc[-1]])

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("contour dissipation plot")
plt.title(fr'$\Phi_1$')
plt.colorbar()
# plt.legend()
plt.show()
plt.savefig('Phi.png')

# %%

# %%
##################################################
# E8
##################################################
new_prob(8)


