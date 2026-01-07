% element degree of freedom vector as specified on the exam (expressed in meters)
ae = [5.1;-11.0;4.0;-11.1;5.0;-10.0;4.0;-10.0]*1e-3;

ae_iso = 0*ae;

ae_iso(1:4) = ae(5:8);
ae_iso(5:6) = ae(3:4);
ae_iso(7:8) = ae(1:2);

% element geometry
a = 1.5;
b = 1.0;

E = 30e9;
nu = 0.2;

dN_dxi = [-1/4 1/4 1/4 -1/4];
dN_deta = [0 -1/2 1/2 0];

xe = [0 a a 0]';
ye = [-b -b 0 0]';

dx_dxi = dN_dxi*xe;
dx_deta = dN_deta*xe;
dy_dxi = dN_dxi*ye;
dy_deta = dN_deta*ye;

J = [dx_dxi dx_deta;dy_dxi dy_deta]

detJ = det(J)

D = hooke(2,E,nu);

D = D([1 2 4],[1 2 4]);

Be_th = inv(J')*[dN_dxi;dN_deta];

Be = zeros(3,8);

Be(1,1:2:end) = Be_th(1,:);
Be(2,2:2:end) = Be_th(2,:);
Be(3,1:2:end) = Be_th(2,:);
Be(3,2:2:end) = Be_th(1,:)

sigma = D*Be*ae_iso


