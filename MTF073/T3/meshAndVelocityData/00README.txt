Solver inputs:

nSIMPLEiter    = 1000           # Maximum number of SIMPLE iterations
nLinSolIter_pp = 10             # Number of linear solver iterations for pp-equation
nLinSolIter_uv = 3              # Number of Gauss-Seidel iterations for u/v-equations
resTol         = 0.001          # Set convergence criteria for residuals
alphaUV        = 0.7            # Under-relaxation factor for u and v
alphaP         = 0.3            # Under-relaxation factor for p
linSol_pp      = 'TDMA'         # Either 'GS' or 'TDMA'
scheme         = 'Hybrid'       # Either 'FOU_CD' or 'Hybrid'
RhieChow       = 'equiCorr'     # Either 'noCorr', 'equiCorr' or 'nonEquiCorr'
pRef_i = 3 # P=0 in some internal node (1..nI-2, not on boundary)
pRef_j = 3 # P=0 in some internal node (1..nJ-2, not on boundary)