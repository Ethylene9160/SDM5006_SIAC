% MRAC based on Lyapunov stability theory when the state is completely measurable

clear all; close all;
h=0.01; L=40/h;                     % Numerical integration step and simulation length
Ap=[0 1;-6 -7]; Bp=[0;8]; 			% Plant parameters
Am=[0 1;-10 -5]; Bm=[0;2]; 			% Reference model parameters
Sz=size(Bp); n=Sz(1); m=Sz(2); 		% State vector, input dimension
P=[3 1;1 1];                        % Positive definite matrix
R1=1*eye(m); R2=1*eye(m); 			% Adaptive parameter matries
F0=zeros(m,n); K0=zeros(m); 		% Initial value
yr0=zeros(m,1); u0=zeros(m,1); e0=zeros(n,1);
xp0=zeros(n,1); xm0=zeros(n,1);

for k=1:L
    time(k)=k*h;
    yr(k)=1*sin(0.01*pi*time(k))+4*sin(0.2*pi*time(k))+sin(1*pi*time(k)); % Input signal 
    % yr(k)=0*exp(0*time(k));
    xp(:,k)=xp0+h*(Ap*xp0+Bp*u0); 	% Calculate xp
    xm(:,k)=xm0+h*(Am*xm0+Bm*yr0); 	% Calculate xm
    
    e(:,k)=xm(:,k)-xp(:,k); 		% e=xm-xp
    F=F0+h*(R1*Bm'* P*e0*xp0'); 	% Adaptive law
    K=K0+h*(R2*Bm'* P*e0*yr0');
    u(:,k)=K*yr(k)+F*xp(:,k); 		% Control input
    
    parae(1,k)=norm(Am-Ap-Bp*F); 	% Deviation of parameter convergence to Am
    parae(2,k)=norm(Bm-Bp*K); 		% Deviation of parameter convergence to Bm
    
    % Update data
    yr0=yr(:,k); u0=u(:,k); e0=e(:,k);
    xp0=xp(:,k); xm0=xm(:,k);F0=F; K0=K;
end;

figure(1); 
subplot(2,1,1); plot(time,xm(1,:),'r-', time,xp(1,:),'b-');
xlabel('t'); ylabel('x_m_1(t)¡¢x_p_1(t)');
legend('x_m_1(t)','x_p_1(t)');
subplot(2,1,2);plot(time,xm(2,:),'r-', time,xp(2,:),'b-');
xlabel('t'); ylabel('x_m_2(t)¡¢x_p_2(t)');
legend('x_m_2(t)','x_p_2(t)');

figure(2);
plot(time,parae(1,:),'r', time,parae(2,:),'b');
xlabel('t'); legend('||A_m-A_p-B_p*F||_2','||B_m-B_p*K||_2');
