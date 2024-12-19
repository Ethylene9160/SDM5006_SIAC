% Adjustable gain Lyapunov-MRAC

clear all; close all;
h=0.1; L=100/h;                          % Numerical integration step and simulation Length				 		%  (reducing h can improve integration accuracy)
num=[2 1]; den=[1 2 1]; n=length(den)-1; % Plant parameters (strictly positive real)
kp=1; 
[Ap,Bp,Cp,Dp]=tf2ss(kp*num,den);        % Plant parameters 
                                        % (transfer function to state space)
km=1; 
[Am,Bm,Cm,Dm]=tf2ss(km*num,den);        % Reference model parameters
gamma=2;                              % Adaptive gain
yr0=0; u0=0; e0=0;                      % Initial value
xp0=zeros(n,1); xm0=zeros(n,1);         % Initial value of the state vector
kc0=0;                                  % Initial value of the adjustable gain
r=2; 

yr=r*[ones(1,L/4) -ones(1,L/4) ones(1,L/4) -ones(1,L/4)]; 	% Input signal
for k=1:L
    time(k)=k*h;
    xp(:,k)=xp0+h*(Ap*xp0+Bp*u0);
    yp(k)=Cp*xp(:,k);                   % Calculate  yp
    
    xm(:,k)=xm0+h*(Am*xm0+Bm*yr0);
    ym(k)=Cm*xm(:,k);                   % Calculate ym
    
    e(k)=ym(k)-yp(k);          			% e=ym-yp
    kc=kc0+h*gamma*e0*yr0;      		% Lyapunov-MRAC adaptive law
    u(k)=kc*yr(k);                      % Control input

    % Update data
    yr0=yr(k); u0=u(k); e0=e(k);xp0=xp(:,k); 
    xm0=xm(:,k);kc0=kc;
end;

subplot(2,1,1);plot(time,yr,'b-.',time,ym,'r', time,yp,'k-');
xlabel('t'); ylabel('y_m(t)¡¢y_p(t)');
legend('y_r(t)','y_m(t)','y_p(t)');
subplot(2,1,2);plot(time,u);
xlabel('t'); ylabel('u(t)');
