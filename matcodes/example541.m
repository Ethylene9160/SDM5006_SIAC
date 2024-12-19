% Narendra adaptive control law (N-m=1, unstable plant)

clear all; close all;
h=0.01; L=40/h;                     % Numerical integration step and simulation length
nump=[1 1]; denp=[1 -5 6]; 
[Ap,Bp,Cp,Dp]=tf2ss(nump,denp); 
n=length(denp)-1;                   % Plant parameters
numm=[1 2]; denm=[1 3 6]; 
[Am,Bm,Cm,Dm]=tf2ss(numm,denm);     % Reference model parameters
Df=numm;                            % Denominator polynomial of the transfer function 
                                    % of the auxiliary signal generator
Af=[[zeros(n-2,1),eye(n-2)];- Df(n:-1:2)]; 	% Auxiliary signal generator state matrix
Bf=[zeros(n-2,1);1];                % Auxiliary signal generator input matrix
yr0=0; yp0=0; u0=0; e0=0;           % Initial values
v10=zeros(n-1,1); v20=zeros(n-1,1); % Initial values of the auxiliary signal generator states
xp0=zeros(n,1); xm0=zeros(n,1);     % Initial values of the state vector
theta0=zeros(2*n,1);                % Initial values of the adjustable parameter vector

r=2; yr=r*[ones(1,L/4) -ones(1,L/4) ones(1,L/4) -ones(1,L/4)]; 	% Reference input signal
Gamma=10*eye(2*n);                  % Adaptive gain matrix (positive definite matrix)

for k=1:L
    time(k)=k*h;
    xp(:,k)=xp0+h*(Ap*xp0+Bp*u0);
    yp(k)=Cp*xp(:,k)+Dp*u0;             % Calculate  yp 
    
    xm(:,k)=xm0+h*(Am*xm0+Bm*yr0);
    ym(k)=Cm*xm(:,k)+Dm*yr0;            % Calculate ym
    
    e(k)=ym(k)-yp(k);       			% e=ym-yp
    v1=v10+h*(Af*v10+Bf*u0); 			% Calculation v1
    v2=v20+h*(Af*v20+Bf*yp0); 			% Calculation v1
    
    phi0=[yr0;v10;yp0;v20]; 			% Set up the data vector at k-1 time
    theta(:,k)=theta0+h*e0*Gamma*phi0; 	% Adaptive law
    phi=[yr(k);v1;yp(k);v2]; 			% Set up the data vector at time k
    u(k)=theta(:,k)'* phi; 				% Adaptive control law

    % Update data
    yr0=yr(k); yp0=yp(k); u0=u(k); e0=e(k);v10=v1; v20=v2;
    xp0=xp(:,k); xm0=xm(:,k);phi0=phi; theta0=theta(:,k);
end;

subplot(2,1,1);plot(time,ym,'r-', time,yp,'b-');xlabel('t'); 
ylabel('y_m(t)¡¢y_p(t)');legend('y_m(t)','y_p(t)');
subplot(2,1,2); plot(time,u); xlabel('t'); ylabel('u(t)');
