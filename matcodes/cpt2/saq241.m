% Recursive generalised least squares parameter estimation (RELS)

clear all; close all;
a=[1 1.8 0.8]'; c=[1 -0.8 -0.2]';  	% Plant parameters
na=length(a)-1;  nc=length(c)-1; 	% na and nc are the order of  A and C

L=1000;                         % Simulation length
yk=zeros(na,1); 				% Initial values of the uutput 
xik=zeros(nc,1); 				% Initial values of the noise
xiek=zeros(nc,1);               % Initial value of noise estimation
xi=sqrt(0.1)*randn(L,1); 		% White noise sequence
theta=[a(2:na+1);c(2:nc+1)]; 	% Plant parameters
thetaE_1=zeros(na+nc,1);        % na+nc is the number of 
                                %       identification parameters
P=10^6*eye(na+nc);
for k=1:L
    phi=[-yk;xik];
    y(k)=phi'*theta+xi(k); 		% Collect output data
    phie=[-yk;xiek];            % Set up phie

   %Recursive augmented least squares method
    K=P*phie/(1+phie'*P*phie);
    thetaE(:,k)=thetaE_1+K*(y(k)-phie'* thetaE_1);
    P=(eye(na+nc)-K*phie')*P;
    xie=y(k)-phie'* thetaE(:,k);  % Estimated value of white noise

    %Update data
    thetaE_1=thetaE(:,k);
    for i=na:-1:2  yk(i)=yk(i-1); end
    yk(1)=y(k);
    for i=nc:-1:2  xik(i)=xik(i-1);xiek(i)=xiek(i-1);end
    xik(1)=xi(k);
    xiek(1)=xie;
end

figure(1);
plot([1:L],thetaE(1:na,:));
xlabel('k'); 
legend('a_1','a_2'); 
axis([0 L -2 2]);

figure(2);
plot([1:L],thetaE(na+1:na+nc,:));
xlabel('k'); 
legend('c_1','c_2'); 
axis([0 L -2 2]);

thetaE(:,L)
