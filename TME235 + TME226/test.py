# %%
from IPython.display import display, Math
import sympy as smp
import sympy.printing as printing


def disp(expr):
    latex_expr = smp.latex(expr)
    return display(Math(latex_expr))


# make a symbol
x = smp.Symbol("x")

# make the derivative of sin(x)*e ^ x
ans1 = smp.diff(smp.sin(x) * smp.exp(x), x)
print("derivative of sin(x)*e ^ x : ", ans1)

# Compute (e ^ x * sin(x)+ e ^ x * cos(x))dx
ans2 = smp.integrate(smp.exp(x) * smp.sin(x) + smp.exp(x) * smp.cos(x), x)
print("indefinite integration is : ", ans2)

# Compute definite integral of sin(x ^ 2)dx
# in b / w interval of ? and ?? .
ans3 = smp.integrate(smp.sin(x**2), (x, -smp.oo, smp.oo))
print("definite integration is : ", ans3)

# Find the limit of sin(x) / x given x tends to 0
ans4 = smp.limit(smp.sin(x) / x, x, 0)
print("limit is : ", ans4)

# Solve quadratic equation like, example : x ^ 2?2 = 0
ans5 = smp.solve(x**2 - 2, x)
disp(ans5)
