%Recursive gradient correction parameter estimation (RGC) for deterministic systems

clear all; 
close all;
a=[1 -1.5 0.7]'; b=[1 0.5]'; d=3; 		% Object parameters
na=length(a)-1; nb=length(b)-1; 		% na and nb are the order of A and B
L=400;                                  % Simulation length
uk=zeros(d+nb,1);                       % Input initial value: uk(i) means u(k-i)
yk=zeros(na,1);                         % Output initial value
u=randn(L,1);                           % Input adopts white noise sequence
theta=[a(2:na+1);b];                    % Object parameter true value
thetae_1=zeros(na+nb+1,1);             % Initial value of parameter estimation
alpha=1;                                % Range (0,2)
c=0.1;                                  % Correction factor

for k=1:L
   phi=[-yk; uk(d:d+nb)];
   y(k)=phi'* theta; 		       		% Collect output data
   thetae(:,k)=thetae_1+alpha*phi*(y(k)-phi'* thetae_1)/(c+phi'* phi); 

   % Recursive gradient correction algorithm
   % Update data
   thetae_1=thetae(:,k);
   for i=d+nb:-1:2  uk(i)=uk(i-1);  end;
   uk(1)=u(k);
   for i=na:-1:2  yk(i)=yk(i-1);  end;
   yk(1)=y(k);
end;
plot([1:L],thetae);  xlabel('k'); 
legend('a_1', 'a_2', 'b_0', 'b_1'); axis([0 L -2 2]);
