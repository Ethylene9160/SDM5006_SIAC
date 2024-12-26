function Out=genPRBS(n,a,del,L);    
% Generate a PRBS signal
% Parameters are 
%    n: the number of registers, 
%    a: the altitude of M-sequence, 
%  del: the clock pulse period, 
%    L: the length of M-sequence to be required 

% Make Out be empty for storing binary sequence
Out = [];      

% Initialize  n registers
for i = 1:n  Reg(i) = 1; end
Out(1)=-a;

for i=2:L
        Rs=Reg(1);
        Reg(1)= xor(Reg(n-2),Reg(n));   % Modulo 2 adder
        j=2;
        while j<=n                      % Registers shift                
             Rs1=Reg(j);
             Reg(j)=Rs;
             j=j+1;
             Rs=Rs1;
        end        
        if (Reg(n)==1)  Out(i) =-a; end
        if (Reg(n)==0)  Out(i)=a; end
end   
