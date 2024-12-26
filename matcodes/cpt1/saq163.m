% Identification of a system impulse response

clc; close all; clear all;

R = 100e3;          	% 100k ohm
C = 1e-6;             	% 1 uf
tc = R*C;               % Time constant

% Generate M-sequence
n=5;
a=1;                % Level of the PRBS   1 --- -a   0 --- +a
Del = 30e-3;      	% Clock pulse period
N=2^n-1;           	% Period of M sequence
L=2*N;             

% Generate M-sequence using the 'genPRBS' function
Out = genPRBS(n,a,Del,L);

% Generate the response y(t) of the system
s = tf('s'); 
G = 100/(s*s+10*s+100)
ft = L*Del;
time = 0:Del:ft-Del;
y = lsim(G, Out, time);

% Plot the input and output of the system
figure
stairs(time,Out);
axis([0 1.0 -2.5 2.5]);
hold on
plot(time,y,'r');
hold off

pause;

% Compute Rxy(i*Del)
temp = 0.0;
Rxy = [];
iDelv=[];
for i=1:N
    tau=i-1;
    iDelv=[iDelv;tau*Del];
       for j=1:N
           temp=temp+sign(Out(j))*y(j+tau);
       end
Rxyi = (a/N)*temp;
temp=0.0;
Rxy = [Rxy;  tau  Rxyi];
end

% Compute ghat & g
Lr = length(Rxy);
C = -Rxy(Lr, 2);
S = (N+1)*a^2*Del/N;
Rxy_iDel = Rxy(:,2);
ghat=(Rxy_iDel+ C )/S;
ghat(1)=2*ghat(1);
szeta=sqrt(1-0.5^2);
g=10*exp(-5.*iDelv)/szeta.*sin(10*szeta.*iDelv); % Impulse response of G(s)
Result = [Rxy  ghat   g];
 
disp(' -------------------------------------------');
disp(' i      Rxy(iDel)     ghat          g');
disp(' -------------------------------------------');
disp(num2str(Result));
disp(' -------------------------------------------');

% Display the results
plot(1:Lr,Result(:,3:4)); 
xlabel('k (step)'); ylabel('Impulse Responses'); legend('ghat(k)','g(k)');
