% Recursive least squares parameter estimation (RLS)

clear all; close all;
a=[1 -1.5 0.7]'; b=[1 0.5]'; d=3; 	% Plant parameters
na=length(a)-1; nb=length(b)-1; 	% na and nb are the order of A and B
L=400;                              % Simulation length
uk=zeros(d+nb,1);                   % Initial values of the input: uk(i) means u(k-i)
yk=zeros(na,1);                     % Initial values of the output: 
u=randn(L,1);                       % Input adopts a white noise sequence
xi=sqrt(0.1)*randn(L,1); 			% White noise sequence
Truetheta=[a(2:na+1);b];            % True values of the plant parameters
thetaE_1=zeros(na+nb+1,1);          % Initial values of thetae
P=10^6*eye(na+nb+1);

for k=1:L 
    Phi=[-yk;uk(d:d+nb)]; 			% Phi is the column vector
    y(k)=Phi'* Truetheta+xi(k); 	% Output data

    % Recursive least squares method
    K=P*Phi/(1+Phi'*P*Phi);
    thetaE(:,k)=thetaE_1+K*(y(k)-Phi'* thetaE_1);
    P=(eye(na+nb+1)-K*Phi')*P;		% Update data
    thetaE_1=thetaE(:,k);
    for i=d+nb:-1:2  uk(i)=uk(i-1);  end;
    uk(1)=u(k);
    for i=na:-1:2  yk(i)=yk(i-1);  end;
    yk(1)=y(k);
end;

plot([1:L],thetaE); 				% line([1,L],[theta,theta]);
xlabel('k'); 
legend('a_ 1','a_2','b_0','b_ 1'); 
axis([0 L -2 1.5]);

thetaE(:,L)


