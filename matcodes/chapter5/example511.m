% Adjustable gain MIT-MRAC

clear all; close all;
h=0.1; L=100/h;                     % Numerical integration step and simulation step
num=[1];
den=[1 1 1]; n=length(den)-1; 		% Plant parameters
kp=1.5; 
[Ap,Bp,Cp,Dp]=tf2ss(kp*num,den); 	% Transfer function to state space 
km=1.5; 
[Am,Bm,Cm,Dm]=tf2ss(km*num,den); 	% Reference model parameters
gamma=0.1;                          % Adaptive gain
yr0=0; u0=0; e0=0; ym0=0; 			% Initial values
xp0=zeros(n,1); xm0=zeros(n,1); 	% Initial values of the state vector
kc0=0;                              % Initial value of the adjustable gain
r=0.6; My=1; 
%r=1.2; My=2;
%r=3.2; My=10;

yr=r*[ones(1,L/4) -ones(1,L/4) ones(1,L/4) -ones(1,L/4)]; 	% Input signal
for k=1:L
    time(k)=k*h;
    xp(:,k)=xp0+h*(Ap*xp0+Bp*u0);
    yp(k)=Cp*xp(:,k)+Dp*u0;         % Calculate yp

    xm(:,k)=xm0+h*(Am*xm0+Bm*yr0);
    ym(k)=Cm*xm(:,k)+Dm*yr0;        % Calculate ym
 
    e(k)=ym(k)-yp(k);               % e=ym-yp
    kc=kc0+h*gamma*e0*ym0;          % MIT adaptive law
    u(k)=kc*yr(k);                  % Control input		

    %Update datay
    yr0=yr(k); u0=u(k); e0=e(k); 
    ym0=ym(k);xp0=xp(:,k); 
    xm0=xm(:,k);
    kc0=kc;
end;

plot(time,yr,'b-',time,ym,'r', time,yp,'k-');
xlabel('t'); ylabel('y_m(t)��y_p(t)');	
axis([0 L*h -My My]);
legend('y_r','y_m(t)','y_p(t)'); 
