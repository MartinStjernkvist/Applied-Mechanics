% nodal coordinates (expressed in meters)
x_1 = -22e-2;
y_1 = 8e-2;
x_2 = -19e-2; 
y_2 = 7e-2;
x_6 = -17e-2;
y_6 = 13e-2;
x_7 = -20e-2;
y_7 = 14e-2;

alpha = 1000;
thick = 1;

Le = sqrt((x_6-x_2)^2+(y_6-y_2));

Kce = alpha*thick*Le/6*[ 2 1; 1 2 ]
