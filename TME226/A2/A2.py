# %%
# %matplotlib widget

from pl_vect import *
from dphidx_dy import *

# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math
import pandas as pd

from IPython.display import display, Math
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation
import scipy.io as sio
from scipy.optimize import fsolve
from matplotlib import rcParams  # for changing default values
import matplotlib.ticker as ticker

import calfem.core as cfc
import calfem.vis_mpl as cfv
import calfem.mesh as cfm
import calfem.utils as cfu

from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.cm as cm

from pathlib import Path


def new_prob(string):
    print_string = (
        "\n--------------------------------------------\n"
        + "Assignment "
        + str(string)
        + "\n--------------------------------------------\n"
    )
    return print(print_string)


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
dpi = 500

# Set the global font sizes
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("figure", figsize=(8, 4))

script_dir = Path(__file__).parent


def fig(fig_name):
    fig_output_file = script_dir / "figures" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches="tight")
    plt.show()
    print("figure name: ", fig_name)


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.1


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.1")

x2_min = x2_2d[:, 0]
x2_max = x2_2d[:, -1]
x1_position = x1_2d[:, 0]

h_x1 = x2_max - x2_min  # Height of the channel at each x1 position

integral_V1_dx2 = np.trapz(v1_2d, x=x2_2d, axis=1)  # Integrate V1 over x2 for each x1

V_b = integral_V1_dx2 / h_x1  # Calculate bulk velocity profile
print("Vb inlet", V_b[0], "m/s")
print("Vb outlet", V_b[-1], "m/s")

plt.figure()
plt.plot(x1_2d[:, 0], V_b, "o-", label="Bulk velocity")
plt.xlabel("$x_1$")
plt.ylabel("Velocity [m/s]")
plt.title("Bulk velocity vs x1 position")
fig("31 Bulk velocity vs x1 position")

# --------------------------------------------
# ...
# --------------------------------------------

integral_P_dx2 = np.trapz(p_2d, x=x2_2d, axis=1)
P_bulk_sim = integral_P_dx2 / h_x1

# Calculating bernoulli pressure based on bulk velocity
rho = 998  # density of water in kg/m^3
P_ref = P_bulk_sim[0]  # reference pressure at inlet
V_ref = V_b[0]  # reference bulk velocity at inlet

P_Bern = P_ref + 0.5 * rho * (V_ref**2 - V_b**2)

# Comparing pressure drops and creating plot
deltaP_sim = P_bulk_sim[0] - P_bulk_sim[-1]
deltaP_bern = P_Bern[0] - P_Bern[-1]
print(f"ΔP_sim: {deltaP_sim:.3e} Pa")
print(f"ΔP_Bernoulli:  {deltaP_bern:.3e} Pa")

plt.figure()
plt.plot(x1_position, P_bulk_sim, "o-", label="STAR-CCM+ bulk pressure")
plt.plot(x1_position, P_Bern, "s--", label="Bernoulli pressure")
plt.xlabel("$x_1$")
plt.ylabel("Pressure [Pa]")
plt.title("Bulk Pressure vs x1 position")
fig("31 Bulk Pressure vs x1 position")

# --------------------------------------------
# ...
# --------------------------------------------

P_dyn = 0.5 * rho * V_b**2

plt.figure()
plt.plot(x1_position, P_dyn, "o-")
plt.xlabel("$x_1$")
plt.ylabel("Pressure [Pa]")
plt.title("Dynamic pressure vs x1 position")
fig("31 Dynamic pressure vs x1 position")


# --------------------------------------------
# ...
# --------------------------------------------

# Given Reynolds number
Re = 37000

def g(f):
    return 1.0 / math.sqrt(f) - 1.930 * math.log10(Re * math.sqrt(f)) + 0.537

# Secant method implementation
def secant(f0, f1, tol=1e-12, maxiter=100):
    for i in range(maxiter):
        g0, g1 = g(f0), g(f1)
        if abs(g1 - g0) < 1e-16:
            raise RuntimeError("Small denominator in secant method")
        # Secant formula
        f2 = f1 - g1 * (f1 - f0) / (g1 - g0)
        if abs(f2 - f1) < tol:
            print(f"Converged in {i+1} iterations")
            return f2
        f0, f1 = f1, f2
    raise RuntimeError("Secant method did not converge")

# Initial guesses
f_guess1 = 0.01
f_guess2 = 0.03

# Solving for fD
FfacD = secant(f_guess1, f_guess2)
print(f"f_D = {FfacD:.8f}")

H = 151.75e-3  # Height of inlet/outlet (meters)
hmax = 50e-3  # Depth of channel (meters)
R = 192.8e-3  # R in meters
L = 9 * hmax
D_h = 2 * H

L = x1_2d[0, 0] - x1_2d[-1, 0]
deltaVbulk = V_b[0] - V_b[-1]

# Calculating pressure drop using Darcy-Weisbach equation
delta_p = FfacD * (L / D_h) * (rho * deltaVbulk**2 / 2)
print(delta_p, "Pa")

# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.2


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.2")

mu = 1.002e-5  # Dynamic viscosity of water at at 20C

dv1_dx2_at_wall = v1_2d[:, 1] / (x2_2d[:, 1] - x2_2d[:, 0])  # Calculating the gradient at the wall using first two points

tau_w = mu * dv1_dx2_at_wall

plt.figure()
plt.plot(x1_2d[:, 0], tau_w, "r-", label="$\\tau_w$")
plt.xlabel("Streamwise position ($x_1$)")
plt.ylabel("Wall Shear Stress ($\\tau_w$) [Pa]")
plt.title("Wall Shear Stress Along the Bottom Surface")
fig("32 Wall Shear Stress Along the Bottom Surface")

# --------------------------------------------
# ...
# --------------------------------------------

column_to_extract = 2  # The third column which contains wall shear stress magnitude

df = pd.read_csv("InternalTableBottom.csv")
wall_shear_stresses_bottom = df.iloc[:, column_to_extract].to_numpy()
# print(f"Data type of wall_shear_stresses_top: {wall_shear_stresses_bottom.dtype}")
# print(f"Data type of V_b: {V_b.dtype}")

df = pd.read_csv("InternalTableTop.csv")
wall_shear_stresses_top = df.iloc[:, column_to_extract].to_numpy()
# print(f"Data type of wall_shear_stresses_top: {wall_shear_stresses_top.dtype}")
# print(f"Data type of V_b: {V_b.dtype}")
# print(np.shape(wall_shear_stresses_bottom))

plt.figure()
plt.plot(
    x1_position, wall_shear_stresses_bottom, "o-", label="Bottom wall shear stress"
)
plt.plot(x1_position, wall_shear_stresses_top, "o-", label="Top wall shear stress")
plt.xlabel("$x_1$")
plt.ylabel("Stress [Pa]")
plt.title("Wall shear stress vs x1 position")
fig("32 Wall shear stress vs x1 position")


Cf_top = np.zeros(len(wall_shear_stresses_top))
Cf_bottom = np.zeros(len(wall_shear_stresses_bottom))

for i in range(len(wall_shear_stresses_top)):
    Cf_top[i] = wall_shear_stresses_top[i] / (0.5 * rho * V_b[i] ** 2)
    Cf_bottom[i] = wall_shear_stresses_bottom[i] / (0.5 * rho * V_b[i] ** 2)

# print(np.shape(Cf_top))

plt.figure()
plt.plot(x1_position, Cf_bottom, "o-", label="Bottom wall skin friction")
plt.plot(x1_position, Cf_top, "o-", label="Top wall skin friction")
plt.xlabel("$x_1$")
plt.ylabel("Skin friction")
plt.title("Skin friction vs x1 position")
fig("32 Skin friction vs x1 position")

# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.3


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.3")

vorticity = dv2dx1_2d - dv1dx2_2d  # omega3 component of vorticity

print("omega3 shape:", vorticity.shape)
print("x1_2d shape:", x1_2d.shape)
print("x2_2d shape:", x2_2d.shape)

plt.figure()
plt.clf()
contour = plt.contourf(x1_2d, x2_2d, vorticity, 100, cmap="RdBu_r")
plt.colorbar(contour, label=r"$\omega_3$ [1/s]")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title(
    r"Vorticity Contours: $\omega_3 = \partial v_2 / \partial x_1 - \partial v_1 / \partial x_2$"
)
plt.axis("equal")
fig("33 Vorticity Contours")

max_i = np.unravel_index(np.argmax(vorticity, axis=None), vorticity.shape)[0]

print("i =", max_i, "Max vorticity at i =", np.max(vorticity))


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.4


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.4")

mu_t = vist_2d * rho  # Kinematic viscosity
ratio = mu_t / mu  # Turbulent viscosity to dynamic viscosity ratio

display(ratio)
print(np.shape(ratio))

plt.figure()
plt.clf()
contour = plt.contourf(x1_2d, x2_2d, ratio, 100, cmap="viridis")
plt.colorbar(contour)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("$\\mu_t / \\mu$")
# plt.axis('equal')
fig("34 Turbulent Viscosity")

# --------------------------------------------
# ...
# --------------------------------------------

plt.figure()
i = 50
plt.plot(ratio[i, :], x2_2d[i, :], "o-", label=f"i={i}, x1={x1_2d[i,0]:.3f}")
i = 100
plt.plot(ratio[i, :], x2_2d[i, :], "o-", label=f"i={i}, x1={x1_2d[i,0]:.3f}")
i = 150
plt.plot(ratio[i, :], x2_2d[i, :], "o-", label=f"i={i}, x1={x1_2d[i,0]:.3f}")
i = 199
plt.plot(ratio[i, :], x2_2d[i, :], "o-", label=f"i={i}, x1={x1_2d[i,0]:.3f}")
plt.ylabel("$x_2$")
plt.xlabel("Skin friction")
plt.title("Skin friction vs x1 position")
fig("34 Skin friction vs x1 position")

# --------------------------------------------
# ...
# --------------------------------------------

column_to_extract = 0  # The first column which contains Ustar

df = pd.read_csv("InternalTableBottom.csv")
Ustar_bottom = df.iloc[:, column_to_extract].to_numpy()

df = pd.read_csv("InternalTableTop.csv")
Ustar_top = df.iloc[:, column_to_extract].to_numpy()

print(mu)

nu = mu / rho  # kinematic viscosity

# Calculating wall-normal distance from bottom wall
x2_wall_dist = x2_2d - x2_2d[:, 0][:, None]  # Reshape for broadcasting (ni x nj)

# Calculating x2+ for bottom wall: x2+ = u_tau * x2 / nu
x2plus_bottom = (Ustar_bottom[:, None] * x2_wall_dist) / nu

indices = [5, 50, 100, 150]  # Indices of x1 stations to plot

# Plotting mu_t/mu versus x2 for selected x1 stations
plt.figure()
for i in indices:
    plt.plot(ratio[i, :], x2_2d[i, :], "-o", label=f"i={i}, x1≈{x1_2d[i,0]:.3f}")
plt.xlabel(r"$\mu_t/\mu$")
plt.ylabel(r"$x_2$ [m]")
plt.title(r"Viscosity ratio $\mu_t/\mu$ vs $x_2$ at selected $x_1$ stations")
fig("34 Viscosity ratio at selected $x_1$ stations")

# Plotting mu_t/mu versus x2+ for selected x1 stations
plt.figure()
for i in indices:
    # sort by x2+ so curve is monotonic
    x2p = x2plus_bottom[i, :].copy()
    y = ratio[i, :].copy()
    order = np.argsort(x2p)
    x2p_s = x2p[order]
    y_s = y[order]
    plt.plot(x2p_s, y_s, "-o", label=f"i={i}, x1={x1_2d[i,0]:.3f}")

plt.xscale("log")
plt.xlabel(r"$x_2^+ = ux_2 / \nu$")
plt.ylabel(r"$\mu_t/\mu$")
plt.title(r"Viscosity ratio $\mu_t/\mu$ vs $x_2^+$ (bottom wall)")
fig("34 Viscosity ratio (bottom wall)")
plt.grid(True, which="both", ls="--", alpha=0.6)


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.5


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.5")

# physical viscosities
nu_t = vist_2d.copy()  # kinematic turbulent viscosity from starccm+

# Calculating the two components of diffusion separately
viscous_component = 2.0 * (nu) * dv1dx1_2d  # 2(ν) * ∂v1/∂x1
turbulent_component = (nu_t) * (dv1dx2_2d + dv2dx1_2d)  # (ν_t) * (∂v1/∂x2 + ∂v2/∂x1)

# Computing divergences
grad_viscous_component, _ = dphidx_dy(
    x1_2d[0:-1, 0:-1], x2_2d[0:-1, 0:-1], viscous_component
)
_, grad_turbulent_component = dphidx_dy(
    x1_2d[0:-1, 0:-1], x2_2d[0:-1, 0:-1], turbulent_component
)

# Contributions and total
viscous_component = grad_viscous_component
turbulent_component = grad_turbulent_component
Diffusion_total = viscous_component + turbulent_component

# Choosing i stations (including hill-top)
i_hill = (np.abs(hmax - x1_2d[:, 1])).argmin()
indices = [i_hill, 50, 150]  # Indices of x1 stations to plot

# Ploting x -y graphs (diffusion vs x2) at selected x1 stations
plt.figure()
for i in indices:
    plt.plot(viscous_component[i, :], x2_2d[i, :], "-", label=f"visc (i={i})")
    plt.plot(turbulent_component[i, :], x2_2d[i, :], "--", label=f"turb(i={i})")
    plt.plot(Diffusion_total[i, :], x2_2d[i, :], ":", label=f"total (i={i})")
plt.xlabel("Diffusion [m/s^2]")
plt.ylabel("$x_2$ [m]")
plt.title("Diff. components vs $x_2$")
fig("35 Diff components vs $x_2$")

# Plotting vs x2+ (bottom wall)
plt.figure()
for i in indices:
    x2p = x2plus_bottom[i, :].copy()
    order = np.argsort(x2p)
    plt.plot(x2p[order], viscous_component[i, :][order], "-", label=f"visc (i={i})")
    plt.plot(x2p[order], turbulent_component[i, :][order], "--", label=f"turb (i={i})")
    plt.plot(x2p[order], Diffusion_total[i, :][order], ":", label=f"total (i={i})")
plt.xscale("log")
plt.xlabel(r"$x_2^+ = ux_2 / \nu$")
plt.ylabel("Diffusion [m/s^2]")
plt.title("Diff. components vs $x_2^+$ at bottom wall")
plt.grid(True, which="both", ls="--", alpha=0.6)
fig("35 Diff components vs $x_2^+$ at bottom wall")


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.6


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.6")

# strain-rate components
s11 = dv1dx1_2d
s22 = dv2dx2_2d
s12 = 0.5 * (dv1dx2_2d + dv2dx1_2d)

# Production: P^k = 2 * nu_t * s_ij s_ij
# Vist_2d is kinematic turbulent viscosity nu_t
Pk = 2.0 * vist_2d * (s11**2 + s22**2 + 2.0 * s12**2)

# contour plots: k, P^k and nu_t

# Plotting k
plt.figure()
cf = plt.contourf(x1_2d, x2_2d, te_2d, 60, cmap="viridis")
plt.colorbar(cf, label="k [J/kg]")
plt.title("Turbulent kinetic energy k")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
fig("36 Turbulent kinetic energy k")

# Plotting P^k
plt.figure()
cf = plt.contourf(x1_2d, x2_2d, np.log(Pk), 60, cmap="inferno")
plt.colorbar(cf, label=r"$P^k$, log scale")
plt.title(r"Production $P^k$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
fig("36 Production $P^k$")

# Plotting nu_t
plt.figure()
cf = plt.contourf(x1_2d, x2_2d, vist_2d, 60, cmap="plasma")
plt.colorbar(cf, label=r"$\nu_t$ [m$^2$/s]")
plt.title(r"Turbulent viscosity $\nu_t$ (kinematic)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
fig("36 Turbulent viscosity")


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.7


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.7")

j_bot = 1
j_top = -3  # -1 is the wall, -2 is the first cell inside

# turbulent kinetic energy k and dissipation eps from STAR-CCM+
k_bot = te_2d[:, j_bot].astype(float)  # turbulent kinetic energy k for bottom wall
eps_ccm_bot = diss_2d[:, j_bot].astype(float)  # dissipation eps for bottom wall

k_top = te_2d[:, j_top].astype(float)  # turbulent kinetic energy k for top wall
eps_ccm_top = diss_2d[:, j_top].astype(float)  # dissipation eps for top wall

# Calculating the distance from the cell center to the wall boundary
x2_bot = np.abs(x2_2d[:, j_bot] - x2_2d[:, 0]).astype(float)
x2_top = np.abs(x2_2d[:, 0] - x2_2d[:, j_top]).astype(float)

# x2_top = np.abs(x2_2d[:, -1] - x2_2d[:, j_top]).astype(float)

# Calculating eps using equation 11.166
eps_model_bot = 2.0 * nu * k_bot / (x2_bot**2)
eps_model_top = 2.0 * nu * k_top / (x2_top**2)

# print(x2_top)
# print(k_top)
# print(eps_model_bot)

x1 = x1_2d[:, 0]  # x1-coordinate for the plot

plt.figure()
plt.plot(x1_position, eps_ccm_bot, "o-", label="$\epsilon_{ccm}$ (Starccm+)")
plt.plot(x1_position, eps_model_bot, "s--", label="$\epsilon_{model}$ (Eq. 11.166)")
plt.xlabel("$x_1$ [m]")
plt.ylabel(r"$\varepsilon$ [m$^2$/s$^3$]")
plt.title("Bottom Wall-Adjacent Cells")
fig("37 Bottom Wall-Adjacent Cells (bottom wall)")

plt.figure()
plt.plot(x1_position, eps_ccm_top, "o-", label="$\epsilon_{ccm}$ (Starccm+)")
plt.plot(x1_position, eps_model_top, "s--", label="$\epsilon_{model}$ (Eq. 11.166)")
plt.xlabel("$x_1$ [m]")
plt.ylabel(r"$\varepsilon$ [m$^2$/s$^3$]")
plt.title("Top Wall-Adjacent Cells")
fig("37 Top Wall-Adjacent Cells (top wall)")

# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


# Assignment 3.9


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob("3.9")

xh2 = np.genfromtxt("xh2.xy", comments="%")
y_2 = xh2[:, 0]  # x_2 coordinates, wall-normal direction.
v1_Exp_2 = xh2[
    :, 1
]  # mean velocity in the streamwise direction (x_1) along wall-normal direction (x_2).

xh3 = np.genfromtxt("xh3.xy", comments="%")
y_3 = xh3[:, 0]
v1_Exp_3 = xh3[:, 1]

xh4 = np.genfromtxt("xh4.xy", comments="%")
y_4 = xh4[:, 0]
v1_Exp_4 = xh4[:, 1]

xh5 = np.genfromtxt("xh5.xy", comments="%")
y_5 = xh5[:, 0]
v1_Exp_5 = xh5[:, 1]

xh6 = np.genfromtxt("xh6.xy", comments="%")
y_6 = xh6[:, 0]
v1_Exp_6 = xh6[:, 1]

xh8 = np.genfromtxt("xh8.xy", comments="%")
y_8 = xh8[:, 0]
v1_Exp_8 = xh8[:, 1]

xh05 = np.genfromtxt("xh05.xy", comments="%")
y_05 = xh05[:, 0]
v1_Exp_05 = xh05[:, 1]

xh005 = np.genfromtxt("xh005.xy", comments="%")
y_005 = xh005[:, 0]
v1_Exp_005 = xh005[:, 1]

plt.figure()
plt.plot(v1_Exp_005, y_005, "bo", label="x/h = 0.05")
plt.plot(v1_Exp_05, y_05, "gs", label="x/h = 0.5")
plt.plot(v1_Exp_1, y_1, "rv", label="x/h = 1")
plt.plot(v1_Exp_2, y_2, "c^", label="x/h = 2")
plt.plot(v1_Exp_3, y_3, "mD", label="x/h = 3")
plt.plot(v1_Exp_4, y_4, "yp", label="x/h = 4")
plt.plot(v1_Exp_5, y_5, "k>", label="x/h = 5")
plt.plot(v1_Exp_6, y_6, "b<", label="x/h = 6")
plt.plot(v1_Exp_8, y_8, "gH", label="x/h = 8")
plt.xlabel("$V_1$")
plt.ylabel("$x_2$")
plt.title("Velocity")
fig("39 Velocity")

# %%
