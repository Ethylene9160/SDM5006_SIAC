% Minimum variance control (MVC)

clear all; close all;
a=[1 -1.7 0.7]; b=[1 0.5]; c=[1 0.2]; d=4;      	% Plant parameters
na=length(a)-1; nb=length(b)-1; nc=length(c)-1; 	% na, nb and nc are the orders 							              
                                                    %    of polynomials A, B and C
nf=nb+d-1;                          % The order of polynomial F
L=400;                              % Control steps
uk=zeros(d+nb,1);                   % Input initial value: uk(i) means u(k-i);
yk=zeros(na,1);                 	% Output initial value
yrk=zeros(nc,1);                	% Expected output initial value
xik=zeros(nc,1); 					% Initial value of a white noise
yr=10*[ones(L/4,1);-ones(L/4,1);ones(L/4,1);-ones(L/4+d,1)]; 	% Expected output
xi=sqrt(0.1)*randn(L,1); 			% White noise sequence
[e,f,g]=sindiophantine(a,b,c,d); 	% Solving a single-step Diophantine equation
                                
for k=1:L  
    time(k)=k;
    y(k)=-a(2:na+1)*yk+b*uk(d:d+nb)+c*[xi(k);xik]; 	% Collect the output data
    u(k)=(-f(2:nf+1)*uk(1:nf)+c*[yr(k+d:-1:k+d-min(d,nc))....	
    yrk(1:nc-d)]-g*[y(k);yk(1:na-1)])/f(1);			% Calculate the control input

% Update data
    for i=d+nb:-1:2
        uk(i)=uk(i-1);
    end;
    uk(1)=u(k);
    for i=na:-1:2
        yk(i)=yk(i-1);
    end;
    yk(1)=y(k);
    for i=nc:-1:2
        yrk(i)=yrk(i-1);
        xik(i)=xik(i-1);
    end;
    if nc>0
        yrk(1)=yr(k);
        xik(1)=xi(k);
    end;
end;

subplot(2,1,1);
plot(time,yr(1:L),'r-',time,y,'b-');
xlabel('k'); ylabel('y_r(k)��y(k)');
legend('y_r(k)','y(k)');
subplot(2,1,2);
plot(time,u);
xlabel('k'); ylabel('u(k)');
