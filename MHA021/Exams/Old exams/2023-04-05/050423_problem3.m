a = 5e-3;

coord = [
     a, 2*a;
   2*a, 2*a;
     0,   a;
     a,   a;
   2*a,   a;
     0,   0;
     a,   0;
   2*a,   0];  

plot(coord(:,1), coord(:,2),'r*')