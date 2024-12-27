% Generation of a white noise sequence and colored noise sequence

clear all; 
close all;
T=600;                                  % Simulation length
D=[1 -1.5 0.7 0.1];                     % Coefficients of D and C polynomials 
C=[1 0.5 0.2];                          % (their roots can be found with the root command)                                       
                                        
nD=length(D)-1; nC=length(C)-1; 		% nD and nC are the order of D and C
xik=zeros(nC,1);                        % Initial value of a white noise, equivalent to ¦Î(k-1),¡­,¦Î(k-nc)
ek=zeros(nD,1);                         % Initial value of colored noise
xi=randn(T,1);                          % randn produces a Gaussian random sequence 
                                        % (white noise sequence) with a mean of 0 and a variance of 1
for k=1:T
    e(k)=-D(2:nD+1)*ek+C*[xi(k);xik];   % Produce colored noise
    for i=nD:-1:2   ek(i)=ek(i-1);  end; 
    ek(1)=e(k);
    for i=nC:-1:2  xik(i)=xik(i-1); end;
    xik(1)=xi(k);
end
subplot(2,1,1);plot(xi);xlabel('k (step)'); ylabel('Noise amplitude'); title('White noise sequence');
subplot(2,1,2);plot(e); xlabel('k (step)'); ylabel('Noise amplitude'); title('Colored noise sequence');
