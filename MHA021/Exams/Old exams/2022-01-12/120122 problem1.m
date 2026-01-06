%% Input data - Do not modify this part
clc, clear variables, close all

phi = 70; % [degrees]
L = 2; % [m]
c = cosd(phi); s = sind(phi); % helper variables

% Element cooridnates
Ex = [0       L*c
      L*c   2*L*c
      2*L*c 3*L*c
      3*L*c 4*L*c
      L*c   3*L*c
      0     3*L*c
      4*L*c   L*c];

Ey = [  0     L*s
       L*s 2*L*s
     2*L*s   L*s
       L*s   0
       L*s   L*s
       0      L*s
       0      L*s ];

eldraw2(Ex, Ey) % Draw the truss structure

E = 210e9;     % Young's modulus [Pa]
A = 1.0e-4;    % cross-sectional area [m^2]
P = 17e3;      % force magnitude [N]
sig_y = 250e6; % material yield limit [Pa] 


%% Write you implementation below
% Anonumous code: 