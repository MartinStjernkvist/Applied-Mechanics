import scipy.io as sio
def read_toplogy_from_mat_file(filename):
    mat_file = sio.loadmat(filename) 
    edof_x=mat_file['Edof'] #note that edof matlab contains one column too much
    edof=edof_x[:,1:13]
    ex=mat_file['Ex']
    ey=mat_file['Ey']
    dof_upper=mat_file['dof_upper']
    dof_lower=mat_file['dof_lower']
    height=mat_file['height']
    ndofs=(mat_file['ndofs']).item() 
    nelem=mat_file['nelem'].item() 
    nnodes=mat_file['nnodes'].item() 
    return ex,ey,edof,dof_upper,dof_lower,ndofs,nelem,nnodes

#example 
filename='topology_coarse_6node.mat'
ex,ey,edof,dof_upper,dof_lower,ndofs,nelem,nnodes=read_toplogy_from_mat_file(filename)
