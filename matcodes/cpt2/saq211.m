% Batch least squares parameter estimation (LS)

clear all;
a=[1 -1.2 0.5]'; b=[1 -0.7 1.5]'; d=2;       % Plant parameters
na=length(a)-1; nb=length(b)-1;         % na and nb are the order of A and B
L=300;                                  % Data length
Uk=zeros(d+nb,1);                       % Initial values of the input: Uk(i) means u(k-i)
Yk=zeros(na,1);                         % Initial values of the output
Reg1=1; Reg2=1; Reg3=1; Reg4=0; Sq=1; 	% Initial values of the shift registers and square wave 
xi= randn(L,1);                         % White noise sequence
xi=0.25*xi/max(abs(xi));
Theta=[a(2:na+1);b];                    % True values of the plant parameter

for k=1:L                               % Phi(k,:) is the row vector, 
    Phi(k,:)=[-Yk;Uk(d:d+nb)]';         %     which is convenient to form phi matrix
    Y(k)=Phi(k,:)*Theta+xi(k);          % Output data
    Ms=xor(Reg3,Reg4);                  % Generate M-sequence
    IMs=xor(Ms,Sq);                     % Generate inverse M-sequence
    if IMs==0  u(k)=-1;  else u(k)=1;  end
    Sq=not(Sq);                         % Generate a square wave 
    
   % Update data
    Reg4=Reg3; Reg3=Reg2; Reg2=Reg1; Reg1=Ms;
    for i=d+nb:-1:2
        Uk(i)=Uk(i-1);
    end
    Uk(1)=u(k);
    for i=na:-1:2
        Yk(i)=Yk(i-1);
    end
    Yk(1)=Y(k);
end;

ThetaE=inv(Phi'* Phi)*Phi'* Y'      % Calculate the parameter estimate thetae
