#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#%%

###########################################################################
# Problem 2
###########################################################################

q0, r, a, b_minus_a, M, E, nu, h = sp.symbols('q0 r a b_minus_a M E nu h', real='True')

D = E * h**3 / (12 * (1 - nu**2))

w = sp.Function('w')(r)

diffeq = sp.Eq(1 / r * sp.diff(r * sp.diff(1 / r * sp.diff(r * sp.diff(w, r), r), r), r), q0 * (r - a)/ (b_minus_a) / D)

w_ = sp.dsolve(diffeq, w).rhs

Mr = D * (-sp.diff(w_, r, 2) - nu * sp.diff(w_, r) / r)
Mphi = D * (-sp.diff(w_, r) / r - nu * sp.diff(w_, r, 2))

V = 1 / r * (sp.diff(r * Mr, r) - Mphi)

bc = [
    w_.subs(r, b_minus_a + a),
    sp.diff(w_, r).subs(r, b_minus_a + a),
    V.subs(r, a),
    Mr.subs(r, a) + M
]

ic = sp.solve(bc, 'C1, C2, C3, C4' )
print('ic: ', ic)
w_sol = w_.subs(ic).simplify()

print(w_sol)

w_func = sp.lambdify((q0, r, a, b_minus_a, M, E, nu, h), w_sol, 'numpy')

r_vals = np.linspace(0.09, 0.3, 300)
vals = w_func(q0=1e6, r=r_vals, a=0.1, b_minus_a=0.2, M=100, E=80e9, nu=0.3, h=0.01)
# value = w_func(1,1,1,1,1,1,1,1)
print(np.argmax(vals))
print(np.argmin(vals))

plt.figure()
plt.plot(r_vals, vals)
plt.show()

#%%

###########################################################################
# Problem 3
###########################################################################

#%%