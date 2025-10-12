%%
clc
close all
clear
%--------------------------------------------------------------------------
ii=200; %Do not change it.
jj=200; %Do not change it.
%--------------------------------------------------------------------------
tt(1)=0;
for j = 2:jj
    tt(j)=tt(j-1)+j;
end

ss=zeros(ii,jj);
ss(:,1)=tt';
for j=1:jj
    for i = 2:ii
        ss(j,i)=ss(j,i-1)+(i-1+(j-1));
    end
end

pp=zeros(ii-1,1);
for i = 1:ii-1
    pp(i) = i^2;
end

subtx=ones(ii,jj);
count=0;
for n = jj:-1:1
    count = jj-n+1;
    subtx(1:count,n)=0;
    subtx(count+1:end,n)=pp(1:end-count+1);
end
gridIndex=ss-subtx;
%--------------------------------------------------------------------------
fileName = sprintf('./output_standard-keps-low-re.csv'); %The name of file extracted from STAR-CCM+
A = importdata(fileName,',');

for i = 1:ii
    for j = 1:jj
        index=gridIndex(i,j);
        u(i,j)=A.data(index+1,2);
        v(i,j)=A.data(index+1,3);
        p(i,j)=A.data(index+1,4);        
        k(i,j)=A.data(index+1,5);
        e(i,j)=A.data(index+1,6);
        mut(i,j)=A.data(index+1,7);
        x(i,j)=A.data(index+1,8);
        y(i,j)=A.data(index+1,9);       
    end
end

u=flip(u');
v=flip(v');
p=flip(p');
k=flip(k');
e=flip(e');
mut=flip(mut');
x=flip(x');
y=flip(y');
%--------------------------------------------------------------------------
ni=200; % Number of cells in x_1 direction, Do not change it.
nj=202; % Number of cells in x_2 direction, Do not change it.
hmax=0.050; % Maximum hill height.
H=3.035*hmax; % Cahnnel height.
L=9*hmax; % Space between two hills summit. 
%--------------------------------------------------------------------------
v1_2d=zeros(ni,nj);
v2_2d=zeros(ni,nj);
p_2d=zeros(ni,nj);
k_2d=zeros(ni,nj);
e_2d=zeros(ni,nj);
mut_2d=zeros(ni,nj);
x1_2d=zeros(ni,nj);
x2_2d=zeros(ni,nj);

v1_2d(:,1)=0;
v1_2d(:,nj)=0;
v1_2d(:,2:nj-1)=u;

v2_2d(:,1)=0;
v2_2d(:,nj)=0;
v2_2d(:,2:nj-1)=v;

p_2d(:,1)=p(:,1);
p_2d(:,nj)=p(:,jj);
p_2d(:,2:nj-1)=p;

k_2d(:,1)=0;
k_2d(:,nj)=0;
k_2d(:,2:nj-1)=k;

e_2d(:,1)=e(:,1);
e_2d(:,nj)=e(:,jj);
e_2d(:,2:nj-1)=e;

mut_2d(:,1)=0;
mut_2d(:,nj)=0;
mut_2d(:,2:nj-1)=mut;

x1_2d(:,1)=x(:,1);
x1_2d(:,nj)=x(:,jj);
x1_2d(:,2:nj-1)=x;

deltaYBottom=(y(:,2)-y(:,1))/2;
deltaYTop=(y(:,jj)-y(:,jj-1))/2;
x2_2d(:,1)=y(:,1)-deltaYBottom;
x2_2d(:,nj)=y(:,jj)+deltaYTop;
x2_2d(:,2:nj-1)=y;

clear u v p k e mut x y
%--------------------------------------------------------------------------
%*************** DO NOT CHANGE ANY PART OF THE ABOVE LINES. ***************
%--------------------------------------------------------------------------
%
%**** LOADING MEASUREMENT DATA AT DIFFERENT X_1 (STREAMWISE) LOCATIONS. ****
%--------------------------------------------------------------------------
load xh1.xy
y_1=xh1(:,1); % x_2 coordinates, wall-normal direction.
v1_Exp_1=xh1(:,2); % mean velocity in the streamwise direction (x_1) along wall-normal direction (x_2). 
v2_Exp_1=xh1(:,3); % mean velocity in the streamwise direction (x_1) along wall-normal direction (x_2). 
uu_Exp_1=xh1(:,4); % Normal Reynolds stress (Re_xx) along wall-normal direction (x_2).  
vv_Exp_1=xh1(:,5); % Normal Reynolds stress (Re_yy) along wall-normal direction (x_2).
uv_Exp_1=xh1(:,6); % Shear Reynolds stress (Re_xy) along wall-normal direction (x_2).
% The locations for the measurement data are: x/h=0.05, 0.5, 1, 2, 3, 4, 5, 6, 7 and 8.
% You should find appropriate "i" corresponds to measurement x locations.
%For example, "xh005.xy", "xh05.xy" and "xh1.xy" are the measurment data at x/h=0.05,x/h=0.5 and x/h=1, repectively.

% conpute velociy gradients
[dv1dx1_2d,dv1dx2_2d] = dphidx_dy(x1_2d,x2_2d,v1_2d,ni,nj);
[dv2dx1_2d,dv2dx2_2d] = dphidx_dy(x1_2d,x2_2d,v2_2d,ni,nj);
%--------------------------------------------------------------------------
% velocity profile plot
%--------------------------------------------------------------------------
figure(1)
xx=hmax;  % choose x=hmax
ii = find(x1_2d(:,1) < xx);
i1=ii(end);
plot(v1_2d(i1,:),x2_2d(i1,:),'b','linewidth',2); %Blue line with line width 2
hold on % Hold the plot 
plot(v1_Exp_1,y_1,'o','linewidth',2);  %plot experiments
handle=gca;
set(handle,'fontsize',20); %20-pt text on the axis
title('Velocity profile','fontsize',[20])
axis([-0.1 0.6 0.0225 H+0.01]) % Set x & y axis
xlabel('V_1','fontsize',20) % 20-pt text
ylabel('x_2','fontsize',20) % 20-pt text
% zoom. 'Position',[left bottom width height]) N.B> (left bottom width height) must have values  between 0 and 1
axes('OuterPosition',[.20 .40 0.45 0.45]) 
plot(v1_2d(i1,:),x2_2d(i1,:),'b','linewidth',2); %Blue line with line width 2
hold on
plot(v1_Exp_1,y_1,'o','linewidth',2);  %plot experiments
axis([-0.1 0.01 0.0225 0.04]) % Set x & y axis

%  print -deps velprof.ps
%--------------------------------------------------------------------------
% Contour plot of v1
%--------------------------------------------------------------------------
figure(2)
surf(x1_2d,x2_2d,v1_2d);
shading interp  % Choose type of shading
view(0,90)      % Look at the plot from the positive z-axis (2D)
axis('equal')   % Make the x and y axis equal, i.e plot the hills in correct proportion
axis([0 L 0 H]) % Zoom-in on the first 0.1m from the inlet
% caxis([-0.0 0.012]) % set color axis [min,max]
handle1=colorbar('EastOutside'); %colorbar location
colormap(jet)
handle=gca;
set(handle,'fontsize',20); % 20-pt text on the colorbar
title('V_1','fontsize',20);

%  print -depsc v1_iso.ps
%--------------------------------------------------------------------------
% plot vector fields
%--------------------------------------------------------------------------
figure(3)
ii=8;
i=1:ii:ni; % plot every ii's i-node
j=1:ii:nj; % plot every ii's j-node
ss=6; %scale vector length
quiver(x1_2d(i,j),x2_2d(i,j),ss*v1_2d(i,j),ss*v2_2d(i,j),ss);
axis('equal')
title('Velocity vector field in XY plane')
xlabel('x')
ylabel('y')
h=gca;
set(h,'fontsi',[20]) %20-pt text

% print -deps vect.ps

