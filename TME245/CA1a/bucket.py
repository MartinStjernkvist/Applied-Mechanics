def generate_deformation_gradient_functions():
    # Fv = sym(Fv,[4,1], real)
    Fv = sp.Matrix(sp.symbols('Fv0:4', real=True)) # Fv0, Fv1, Fv2, Fv3
    
    Gmod, lam = sp.symbols('Gmod lambda', real=True)

    # MATLAB: F=[Fv(1) Fv(3) 0; Fv(4) Fv(2) 0; 0 0 1];
    # Mapping indices: 1->0, 2->1, 3->2, 4->3
    F = sp.Matrix([
        [Fv[0], Fv[2], 0],
        [Fv[3], Fv[1], 0],
        [0,     0,     1]
    ])

    # MATLAB: C=F*F
    C = F * F 
    
    # MATLAB: invC=simplify(inv(C))
    invC = sp.simplify(C.inv())
    
    # MATLAB: J=det(F)
    J = F.det()

    # MATLAB: S=Gmod*( eye(3)-invC)+lambda*log(J)*invC;
    S = Gmod * (sp.eye(3) - invC) + lam * sp.log(J) * invC
    
    # MATLAB: P=F*S
    P = F * S

    # MATLAB: Pv=[P(1,1) P(2,2) P(1,2) P(2,1)]
    # Python indices: (0,0), (1,1), (0,1), (1,0)
    Pv = sp.Matrix([P[0,0], P[1,1], P[0,1], P[1,0]])

    P_NH_func = sp.lambdify((Fv, Gmod, lam), Pv, modules='numpy')

    # MATLAB: dPvdFv=sym(dPvFv,[4,4], real )
    # Loop i=1:4 ... gradient(Pv(i),Fv)
    
    # In SymPy, we can calculate the Jacobian matrix directly without a loop
    dPvdFv = Pv.jacobian(Fv)

    dPdF_NH_func = sp.lambdify((Fv, Gmod, lam), dPvdFv, modules='numpy')

    return P_NH_func, dPdF_NH_func