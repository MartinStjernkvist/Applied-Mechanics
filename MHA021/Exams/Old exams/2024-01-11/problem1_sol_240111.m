% FE-exam in MHA064 and VSM167 2024-01-11

% Problem 1

% b)  x1 corresponds to x_i and x2 to x_{i+1} repsectively
syms x x1 x2 q0 Le L A0 E

N1 = (x2 - x)/(x2-x1)
N2 = (x - x1)/(x2-x1)
Ne = [N1, N2]
q(x) = q0*(1-0.2*x/L)

% Load vector
fle = int(transpose(Ne)*q, x1, x2)



% Stiffness matrix 
Be = [-1/Le, 1/Le]
A(x) = A0/2*(2 - (x/L)^2)

I = int(A, x1, x2)
Ke = E*transpose(Be) * Be * I

% Convert the symbolic expressions into functions
Ke_ = matlabFunction(Ke)
fle_ = matlabFunction(fle)

% c) solve 3 element problem with the following numerical values
L = 1;
A0 = 20e-4;
E = 80e9;
q0 = 100e3;
Le = L/3

K1 = Ke_(A0, E, L, Le,    0,   Le)
K2 = Ke_(A0, E, L, Le,   Le, 2*Le)
K3 = Ke_(A0, E, L, Le, 2*Le, 3*Le)

fl1 = fle_(L, q0,    0,   Le)
fl2 = fle_(L, q0,   Le, 2*Le)
fl3 = fle_(L, q0, 2*Le, 3*Le)

% Assemble equations 
K = zeros(4,4); fl = zeros(4,1);
[K, fl] = assem([1  1 2], K, K1, fl, fl1)
[K, fl] = assem([2  2 3], K, K2, fl, fl2)
[K, fl] = assem([3  3 4], K, K3, fl, fl3)

bc = [1 0] % clamped to the left
a = solveq(K, fl, bc)

% displacement at the right end
a(4)

