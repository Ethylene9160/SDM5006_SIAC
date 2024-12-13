% Identification using the least squares method, based on Simulink data
% dataout.data is obtained from Simulink 

% Display the data from the Simulink
Ts=0.1;                         % The sampling period
Time=[0:1:size(dataout.data(:,1))-1]*Ts;
plot(Time,dataout.data);

% Initialize the parameters
na=1;
nb=1;
n=max(na,nb);              		% Depth of historical data
N=size(dataout.data); 			% The ¡®dataout¡± stores the input-output data
N=N(1);                         % Total number of historical data
phph=zeros(na+nb);
phy=zeros(na+nb,1);
for t=n+1:N     				% The loop part
    tmp=[];
    for j=1:na tmp=[tmp,-dataout.data(t-j,1)]; end;
    for j=1:nb tmp=[tmp,dataout.data(t-j,2)]; end;
    phph=phph+tmp'*tmp;    
    phy=phy+tmp'*dataout.data(t,1);
end;

% Estimate¦Èof the discrete model
theta=inv(phph)*phy			 	

% Coverte the discrete model to a continuous model
sysd=tf(theta(2), [1 theta(1)],Ts);
d2c(sysd)