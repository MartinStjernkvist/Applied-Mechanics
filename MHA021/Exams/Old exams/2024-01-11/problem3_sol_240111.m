%% Problem 3b
clc, clear variables, close all

xe = [3; 4; 4; 3]*1e-2;
ye = [0; 0; 1; 0.8]*1e-2;
rho = 7800;
g = 9.81;
omega = 20;

% Gauss integration
% three points in each direction
% Xi = [-0.774597, 0, 0.774597];
% Eta = [-0.774597, 0, 0.774597];
% H = [0.55556 0.88889 0.55556];

% Xi = [-1/sqrt(3) 1/sqrt(3)];
% Eta = [-1/sqrt(3) 1/sqrt(3)];
% H = [1 1];
% 
Xi = [0];
Eta = [0];
H = [2];

fe = zeros(8,1);
area = 0;
for i=1:length(Xi)
    for j = 1:length(Xi)
        xi = Xi(i);
        eta = Eta(j);
        [Ne, Be, detJ] = compute_N_B(xi, eta, xe, ye);
        x = Ne(1,1:2:end)*xe; % x-value in integration point
        fe = fe + Ne' * [rho*g*x; -rho*g] * detJ * H(i) * H(j);
        area = area + detJ*H(i)*H(j);
    end
end
fe
area


function [Ne, Be, detJ] = compute_N_B(xi, eta, xe, ye)

N1 = 1/4*(xi-1)*(eta-1);
N2 = -1/4*(xi+1)*(eta-1);
N3 = 1/4*(xi+1)*(eta+1);
N4 = -1/4*(xi-1)*(eta+1);
Ne = [N1, 0, N2, 0, N3, 0, N4, 0;0, N1, 0, N2, 0, N3, 0, N4];

dN1_dxi = 1/4*(eta-1);
dN2_dxi = -1/4*(eta-1);
dN3_dxi = 1/4*(eta+1);
dN4_dxi = -1/4*(eta+1);

dN1_deta = 1/4*(xi-1);
dN2_deta = -1/4*(xi+1);
dN3_deta = 1/4*(xi+1);
dN4_deta = -1/4*(xi-1);

dN_dxi = [dN1_dxi dN2_dxi dN3_dxi dN4_dxi];
dN_deta = [dN1_deta dN2_deta dN3_deta dN4_deta];

dx_dxi = dN_dxi*xe
dx_deta = dN_deta*xe
dy_dxi = dN_dxi*ye
dy_deta = dN_deta*ye

J = [dx_dxi, dx_deta;
    dy_dxi, dy_deta];

detJ = det(J)

Be_ = inv(J')*[dN_dxi;
    dN_deta];

Be = zeros(3,8);
Be(1,1:2:end) = Be_(1,:);
Be(2,2:2:end) = Be_(2,:);
Be(3,1:2:end) = Be_(2,:);
Be(3,2:2:end) = Be_(1,:);

end
