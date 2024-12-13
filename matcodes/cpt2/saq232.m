% Forgetting factor recursive least squares parameter estimation (FFRLS)

clear all; close all;
a=[1 -1.5 0.7 0]'; b=[0 1 0.5]'; d=2; 	% Plant parameters
na=length(a)-1; nb=length(b)-1;         % na and nb are the order of A and B
L=1000;                                 % Simulation length
uk=zeros(d+nb,1);                       % Initial values of the input: uk(i) means u(k-i)
yk=zeros(na,1);                         % Initial values of the output
u=randn(L,1);                           % Input adopts a white noise sequence
xi=sqrt(0.1)*randn(L,1);                % White noise sequence
thetae_1=zeros(na+nb+1,1);              % Initial values of thetae
P=10^6*eye(na+nb+1);
lambda=0.98;                            % Forgetting factor range [0.9 1]

for k=1:L
if k==501
    a=[1 -1 0.4 0]'; b=[0 1.5 0.2]'; 	% Plant parameters
end;
    theta(:,k)=[a(2:na+1);b];           % Plant parameter true values
    phi=[-yk;uk(d:d+nb)];
    y(k)=phi'* theta(:,k)+xi(k);        % Output data
    
  % Forgetting factor recursive least square method
    K=P*phi/(lambda+phi'* P*phi);
    thetae(:,k)=thetae_1+K*(y(k)-phi'* thetae_1);
    P=(eye(na+nb+1)-K*phi')* P/lambda;	% Update data
    thetae_1=thetae(:,k);
    for i=d+nb:-1:2  uk(i)=uk(i-1);end
    uk(1)=u(k);
    for i=na:-1:2  yk(i)=yk(i-1);end
    yk(1)=y(k);
end;

subplot(1,2,1); plot([1:L],thetae(1:na,:)); hold on; 
plot([1:L],theta(1:na,:),'k:');xlabel('k'); legend('a_1','a_2','a_3'); axis([0 L -2 2]);

subplot(1,2,2); plot([1:L],thetae(na+1:na+nb+1,:)); hold on; 
plot([1:L],theta(na+1:na+nb+1,:), 'k:');xlabel('k'); legend('b_0','b_1','b_2'); axis([0 L -0.5 2]);

thetae(:,L)

