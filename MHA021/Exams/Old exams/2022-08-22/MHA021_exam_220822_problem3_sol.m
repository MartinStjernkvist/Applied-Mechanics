%% Problem 3b
clc, clear variables, close all

xe = [150; 190; 200.0]*1e-3;
ye = [140; 110; 180]*1e-3;


eldraw2(xe',ye')

t = 10e-3; % thickness
alpha = 12e-6; % coeff of thermal exp.
T0 = 15; % ref. temperature
D = hooke(1, 210e9, 0.3);

Te = [45; 27; 64]; % counter clockwise order - node 16, 78, 43

% Gauss integration 
% three points
Xi = [1/6, 2/3, 1/6];
Eta = [1/6, 1/6, 2/3];
W = [1/6, 1/6, 1/6];

% one point (gives the same since the inregrand is linear
% Xi = [1/3];
% Eta = [1/3];
% W = [1/2];

fe = zeros(6,1);
for ip = 1:length(Xi)
    xi = Xi(ip);
    eta = Eta(ip);
    [Ne, Be, detJ] = compute_N_B(xi, eta, xe, ye);
    T = Ne*Te; % temperature in integration point
    DeltaT = T - T0;
    dA = detJ * W(ip)
    fe = fe + t * alpha * DeltaT * Be' * D * [1; 1; 0] * dA;
end
fe


function [Ne, Be, detJ] = compute_N_B(xi, eta, xe, ye)

    N1 = 1 - xi - eta;
    N2 = xi;
    N3 = eta;
    Ne = [N1, N2, N3];
    
    dN1_dxi = -1;
    dN2_dxi = 1;
    dN3_dxi = 0;

    dN1_deta = -1;
    dN2_deta = 0;
    dN3_deta = 1;

    dN_dxi = [dN1_dxi dN2_dxi dN3_dxi];
    dN_deta = [dN1_deta dN2_deta dN3_deta];

    dx_dxi = dN_dxi*xe;
    dx_deta = dN_deta*xe;
    dy_dxi = dN_dxi*ye;
    dy_deta = dN_deta*ye;

    J = [dx_dxi, dx_deta;
         dy_dxi, dy_deta];
    detJ = det(J);

    Be_ = inv(J')*[dN_dxi;
                   dN_deta];

    Be = zeros(3,6);
    Be(1,1:2:end) = Be_(1,:);
    Be(2,2:2:end) = Be_(2,:);
    Be(3,1:2:end) = Be_(2,:);
    Be(3,2:2:end) = Be_(1,:);


end
