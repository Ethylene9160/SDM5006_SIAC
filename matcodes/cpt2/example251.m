% Parameter estiamtion of multi-variable systems using the LS method

clear all;
close all;
clc

L=15;
U1=zeros(2,1);U2=zeros(2,1);
Y1=zeros(2,1);Y2=zeros(2,1);

% Generate input-output data
A1=[0.5,-0.2;-0.3,0.6];A2=[1.2,-0.6;0.1,-0.6];
B0=[1.0,0.0;0.0,1.0];B1=[0.5,-0.4;0.2,-0.3];B2=[0.3,-0.4;-0.2,0.1];
randn('seed',100)
V=sqrt(0.05)*randn(2,L);     % Noise
randn('seed',1000)
U=randn(2,L);               % Input signal
for k=1:1:L
   time(k)=k;
   Y(1:2,k)=-A1*Y1-A2*Y2+B0*U(1:2,k)+B1*U1+B2*U2+0.5*V(1:2,k);
   U2=U1;U1=U(1:2,k);Y2=Y1;Y1=Y(1:2,k);
end
figure(1);
plot(time,U(1,:),'k',time,U(2,:),'b');
xlabel('Time'),ylabel('Input');
figure(2)
plot(time,Y(1,:),'k',time,Y(2,:),'b');
xlabel('Time'),ylabel('Output');
clear time

% Identification using the LS method
for i=3:L
    % The identification coefficient matrix and observation vector
    H(i-2,1:10)=[-Y(1,i-1),-Y(2,i-1),-Y(1,i-2),-Y(2,i-2),U(1,i),U(2,i),U(1,i-1),U(2,i-1),U(1,i-2),U(2,i-2)];
    Y1(i-2)=Y(1,i);
    Y2(i-2)=Y(2,i);
end
theat1=inv(H'*H)*H'*Y1;
theat2=inv(H'*H)*H'*Y2;

% Separation of the parameters
A1H=[theat1(1),theat1(2);theat2(1),theat2(2)]
A1
A2H=[theat1(3),theat1(4);theat2(3),theat2(4)]
A2
B0H=[theat1(5),theat1(6);theat2(5),theat2(6)]
B0
B1H=[theat1(7),theat1(8);theat2(7),theat2(8)]
B1
B2H=[theat1(9),theat1(10);theat2(9),theat2(10)]
B2
