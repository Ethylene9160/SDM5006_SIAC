% Generation of an M-sequence and inverse M-sequence

clear all; close all;
L=60;                           % M-sequence length
Reg1=1; Reg2=0; Reg3=1; Reg4=1; % Shift register initial values 
Reg5=1;
Sq=0;                           % Initial value of the square wave

for k=1:L
    Ms(k)=xor(Reg1,Reg3); 		% Perform XOR operation to generate an M-sequence
    Ms(k)=xor(Ms(k),Reg5); 	   
    IMs=xor(Ms(k),Sq);          % Perform XOR operation to generate an inverse M-sequence
    if IMs==0
        Out(k)=-1;
      else
        Out(k)=1;
    end
Sq=not(Sq);                     % Generate a square wave
Reg5=Reg4; Reg4=Reg3; Reg3=Reg2; Reg2=Reg1; Reg1=Ms(k);    % The registers shift
end

subplot(2,1,1);stairs(Ms); grid;axis([0 L/2 -0.5 1.5]); 
xlabel('k'); ylabel('M-sequence amplitude'); title('M-sequence');
subplot(2,1,2);stairs(Out); grid;axis([0 L -1.5 1.5]); 
xlabel( 'k'); ylabel('Inverse M-sequence amplitude'); title('Inverse M-sequence');
