% Generalized minimum variance control (explicit control law)

clear all; close all;
a=[1 -1.7 0.7]; b=[1 2]; c=[1 0.2]; d=4;   % Plant parameters (no steady-state error)
% a=[1 -2 0.7]; b=[1 2]; c=[1 0.2]; d=4;   % Plant parameters (with steady-state error)
na=length(a)-1; nb=length(b)-1; 
nc=length(c)-1;                 % The order of polynomials A, B and C
nf=nb+d-1; ng=na-1;				% The order of polynomials F and G
P=1; R=1; Q=20; 					% Weighted polynomial Q=0.5, 2, 20;
np=length(P)-1; nr=length(R)-1; nq=length(Q)-1;
L=400;                          % Control steps
uk=zeros(d+nb,1); 				% Input initial value: uk (i) means u (k-i);
yk=zeros(na,1); 				% Output initial value
yrk=zeros(nc,1); 				% Expected output initial value
xik=zeros(nc,1); 				% Initial value of a white noise
yr=10*[ones(L/4,1);-ones(L/4,1);ones(L/4,1);-ones(L/4+d,1)]; % Expected output
xi=sqrt(0.1)*randn(L,1); 					% White noise sequence

[e,f,g]=sindiophantine(a,b,c,d); 			% Solving single-step Diophantine equation
CQ=conv(c,Q); FP=conv(f,P); CR=conv(c,R); GP=conv(g,P);      % CQ=C*Q

for k=1:L
    time(k)=k;
    y(k)=-a(2:na+1)*yk+b*uk(d:d+nb)+c*[xi(k);xik];           % Collect output data
    u(k)=(-Q(1)*CQ(2:nc+nq+1)*uk(1:nc+nq)/b(1)-FP(2:np+nf+1)*uk(1:np+nf)...
            +CR*[yr(k+d:-1:k+d-min(d,nr+nc)); yrk(1:nr+nc-d)]...
            -GP*[y(k); yk(1:np+ng)])/(Q(1)*CQ(1)/b(1)+FP(1));% Calculate the control input
   
    % Update data
    for i=d+nb:-1:2 uk(i)=uk(i-1);end;
    uk(1)=u(k);
    for i=na:-1:2 yk(i)=yk(i-1);end
    yk(1)=y(k);
    for i=nc:-1:2 yrk(i)=yrk(i-1);xik(i)=xik(i-1);end
    if nc>0 yrk(1)=yr(k);xik(1)=xi(k);end
end;

subplot(2,1,1);plot(time,yr(1:L), 'r-', time,y,'b-');
xlabel('k'); ylabel('y_r(k), y(k)');
legend('y_r(k)','y(k)');
subplot(2,1,2);
plot(time,u);
xlabel('k'); ylabel('u(k)');
