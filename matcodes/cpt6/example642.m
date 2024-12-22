% Pole placement indirect self-tuning control

clear all; close all;
a=[1 -1.6065 0.6065]; b=[0.1065 0.0902]; c=[1 0.5]; d=3; 	% Plant parameters
Am=[1 -1.3205 0.4966]; 				% Expected closed-loop characteristic polynomial
na=length(a)-1; nb=length(b)-1; nc=length(c)-1; 		% The orders of polynomials A, B and C
nam=length(Am)-1; 					% The order of Am
nf=nb+d-1; ng=na-1;L=400;			% Control steps
uk=zeros(d+nb,1); 					% Input initial value: uk (i) means u (k-i)
yk=zeros(na,1); 					% Output initial value
yrk=zeros(na,1); 					% Expected output initial value
xik=zeros(nc,1); 					% Initial value of white noise
xiek=zeros(nc,1); 					% Initial value of white noise estimation
yr=10*[ones(L/4,1);-ones(L/4,1);ones(L/4,1);-ones(L/4+d,1)]; 	% Expected output
xi=sqrt(0.01)*randn(L,1); 			% White noise sequence %RELS initial value
thetae_1=0.001*ones(na+nb+1+nc,1);
P=10^6*eye(na+nb+1+nc);lambda=1; 	% Forgetting factor [0.9 1]

for k=1:L
    time(k)=k;
    y(k)=-a(2:na+1)*yk+b*uk(d:d+nb)+c*[xi(k);xik]; 		% Collect output data
    
    % Recursive augmented least squares method
    phie=[-yk(1:na);uk(d:d+nb);xiek];
    K=P*phie/(lambda+phie'* P*phie);
    thetae(:,k)=thetae_1+K*(y(k)-phie'* thetae_1);
    P=(eye(na+nb+1+nc)-K*phie')* P/lambda;
    xie=y(k)-phie'*thetae(:,k); 	% Estimated value of white noise 

    % Extract identification parameters
    ae=[1 thetae(1:na,k)']; be=thetae(na+1:na+nb+1,k)'; 
    ce=[1 thetae(na+nb+2:na+nb+1+nc,k)'];
    if nc>0 if abs(ce(2))>0.8 ce(2)=sign(ce(2))*0.8; end; end;

    % Decomposition of polynomial B
    br=roots(be); 				% Find the roots of B
    b0=be(1); b1=1; 			% B0 is B -; B1 is B+
    Val=0.9; 

    % Determine whether zeros of B is cancelled by modifying the critical value 
    % (if the absolute value of zeros is less than the critical value, it will be cancelled) 
    % decomposition B -, B+
    for i=1:nb if abs(br(i))>=Val b0=conv(b0,[1 -br(i)]);else b1=conv(b1,[1 -br(i)]); end; end;
    Bm1=sum(Am)/sum(b0); 		% Determine polynomial Bm %Determine polynomial A0%
    A0=ce;                      % Can be A0=C
    na0=2*na-1-nam-(length(b1)-1);               % Lowest order of the observer
    A0=1;for i=1:na0 A0=conv(A0,[1 0.5]); end;   % Generate the observer

    % Solve the Diophantine equation to obtain F, G, R
    [F1,G]=diophantine(ae,b0,d,A0,Am);
    F=conv(F1,b1); R=Bm1*A0;nr=length(R)-1;
    
    u(k)=(-F(2:nf+1)*uk(1:nf)+R*[yr(k+d:-1:k+d-min(d,nr));
    yrk(1:nr-d)]-G*[y(k);yk(1:ng)])/F(1);	  % Calculate the  control input
    thetae_1=thetae(:,k);
    
    for i=d+nb:-1:2uk(i)=uk(i-1);end;
    uk(1)=u(k);
    for i=na:-1:2 yk(i)=yk(i-1);yrk(i)=yrk(i-1);end;
    yk(1)=y(k);yrk(1)=yr(k);
    for i=nc:-1:2 xik(i)=xik(i-1);xiek(i)=xiek(i-1);end;
    if nc>0 xik(1)=xi(k);xiek(1)=xie;end;
end;

figure(1);subplot(2,1,1);plot(time,yr(1:L),'r-', time,y,'b-');
xlabel('k'); ylabel('y_r(k)¡¢y(k)');
legend('y_r(k)','y(k)'); axis([0 L -20 20]);
subplot(2,1,2);plot(time,u);
xlabel('k'); ylabel('u(k)'); axis([0 L -40 40]);

figure(2);subplot(211);plot([1:L],thetae(1:na,:),[1:L],thetae(na+nb+2:na+nb+1+nc,:));
xlabel('k'); ylabel('Parameter estimation a, c');
legend('a_1','a_2','c_1'); axis([0 L -2 2]);
subplot(212);plot([1:L],thetae(na+1:na+nb+1,:));
xlabel('k'); ylabel('Parameter estimation b');
legend('b_0','b_1'); axis([0 L 0 0.15]);

