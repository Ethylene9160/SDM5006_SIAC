%Recursive stochastic Newton parameter estimation (RSNA)

clear all; close all;
a=[1 -1.5 0.7]'; b=[1 0.5]'; d=3; 		% Object parameters
na=length(a)-1; nb=length(b)-1; 		% Na and nb are of order A and B
L=400;                                  % Simulation length
uk=zeros(d+nb,1);                       % Input initial value: uk(i) means u(k-i)
yk=zeros(na,1);                         % Output initial value
xik=zeros(na,1);                        % Initial value of white noise ¦Î
etak=zeros(d+nb,1);                     % Initial value of white noise ¦Ç
u=randn(L,1);                           % Input adopts white noise sequence
xi=sqrt(0.1)*randn(L,1);                % White noise sequence ¦Î
eta=sqrt(0.25)*randn(L,1);              % White noise sequence ¦Ç
theta=[a(2:na+1);b];                    % Object parameter true value
thetae_1=zeros(na+nb+1,1);              % Initial value of parameter estimation
Rk_1=eye(na+nb+1);

for k=1:L
    phi=[-yk;uk(d:d+nb)];
    e(k)=a'* [xi(k);xik]-b'* etak(d:d+nb);
    y(k)=phi'* theta+e(k); 				% Collect output data

    %Random Newton algorithm
    R=Rk_1+(phi*phi'- Rk_1)/k;
    dR=det(R);
    if abs (dR)<10^(-6)					% to avoid nonsingularity of matrix R
       R=eye(na+nb+1);
    end;
    IR=inv(R);
    thetae(:,k)=thetae_1+IR*phi*(y(k)-phi'* thetae_1)/k;      %Update data
    thetae_1=thetae(:,k);
    Rk_1=R;
     for i=d+nb:-1:2
        uk(i)=uk(i-1);
        etak(i)=etak(i-1);
    end;
    uk(1)=u(k);
    etak(1)=eta(k);
    for i=na:-1:2
        yk(i)=yk(i-1);
        xik(i)=xik(i-1);
    end;
    yk(1)=y(k); xik(1)=xi(k);
end;
plot([1:L],thetae); xlabel('k'); 
legend('a_1','a_2','b_0','b_1'); 
axis([0 L -2 1.5]);
   
