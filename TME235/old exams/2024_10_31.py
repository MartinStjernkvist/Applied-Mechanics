#%%
import numpy as np
import sympy as sp

#%%

# Problem 2

q0, r, a, b_minus_a, M, D, E, nu = sp.symbols('q0 r a b_minus_a M D E nu', real='True')

w = sp.Function('w')(r)

diffeq = sp.Eq(1 / r * sp.diff(r * sp.diff(1 / r * sp.diff(r * sp.diff(w, r), r), r), r), q0 * (r - a)/ (b_minus_a))

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

w_func = sp.lambdify((q0, r, a, b_minus_a, M, D, E, nu), w_sol, 'numpy')

value = w_func(q0=1, r=1, a=1, b_minus_a=1, M=1, D=1, E=1, nu=1)
# value = w_func(1,1,1,1,1,1,1,1)
print(value)

#%%
