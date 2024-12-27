% Generate a PRBS signal

clear all;
n=3;                                % the number of registers
L=8+1;                              % the length of M-sequence to be required 

Reg=[1 0 1];                        % Initialize n registers
Out(1)=Reg(n);

for i=2:L
        temp=Reg(1);
        Reg(1)=xor(Reg(1),Reg(3));  % Modulo 2 adder
        j=2;
        while j<=n                  % Registers shift                
             temp1=Reg(j);
             Reg(j)=temp;
             j=j+1;
             temp=temp1;
        end        
        Out(i)=Reg(n);
end   

stairs(Out); axis([1 L -0.5 1.5]);  % Plot the PRBS signal


