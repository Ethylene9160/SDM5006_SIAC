% Recursive identification, based on Simulink data
% dataout.data is obtained from Simulink 

% Display the input-output data from the Simulink
Ts=0.1;                         % The sampling period
N=max(size(dataout.data(:,1)));
Time=[0:1:N-1]*Ts;
plot(Time,dataout.data);

% Initialization
na=2;
nb=1;
Nb=nb+1;
n=max(na,Nb);
lambda=0.9;
 
theta=zeros(na+Nb,1);
L=zeros(na+Nb,1);
P=eye(na+Nb);
theta_k=zeros(na+Nb,n);

% Online identification
for i=1+n:N
    phi=[];
    for j=1:na
        phi=[phi,-dataout.data(i-j,1)];
    end
    for j=1:Nb
        phi=[phi,dataout.data(i-j,2)];
    end
    phi=phi';   
    L=P*phi/(lambda+phi'*P*phi);
    theta=theta+L*(dataout.data(i,1)-phi'*theta);
    P=(P-P*phi*phi'*P/(lambda+phi'*P*phi))/lambda;
    
    theta_k(:,i)=theta;
end

plot([1:N],theta_k);xlabel('k'); legend('a_1','a_2','b_0','b_1'); axis([0 N -2 2]);

% Estimation¦Èof the discrete model
theta

% Coverte the discrete model to a continuous model
sysd=tf(theta(3:4)', [1 theta(1:2)'],Ts);
d2c(sysd)
