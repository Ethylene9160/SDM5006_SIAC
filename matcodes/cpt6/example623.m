% Minimum variance direct self-tuning control

clear all; close all;
a=[1 -1.7 0.7]; b=[1 0.5]; c=[1 0.2]; d=4; 	% plant parameters
na=length(a)-1; nb=length(b)-1; nc=length(c)-1;  % The order of polynomials A, B and C
nf=nb+d-1; ng=na-1;			    % The order of polynomials F and G
L=400;                          % Control steps
uk=zeros(d+nf,1); 				% Input initial value: uk (i) means u (k-i);
yk=zeros(d+ng,1);               % Output initial value
yek=zeros(nc,1);                % Initial value of optimal output prediction estimation
yrk=zeros(nc,1);                % Expected output initial value
xik=zeros(nc,1);                % Initial value of a white noise
yr=10*[ones(L/4,1);-ones(L/4,1);ones(L/4,1);-ones(L/4+d,1)];        % Expected output
xi=sqrt(0.1)*randn(L,1);        % White noise sequence
thetaek=zeros(na+nb+d+nc,d);P=10^6*eye(na+nb+d+nc); %Recursive estimation initial value

for k=1:L
    time(k)=k;y(k)=-a(2:na+1)*yk(1:na)+b*uk(d:d+nb)+c*[xi(k);xik]; 	% Collect output data 
    phie=[yk(d:d+ng);uk(d:d+nf);-yek(1:nc)]; 	   % Recursive generalised least squares method
    K=P*phie/(1+phie'* P*phie);
    thetae(:,k)=thetaek(:,1)+K*(y(k)-phie'* thetaek(:,1));
    P=(eye(na+nb+d+nc)-K*phie')* P;
    ye=phie'* thetaek(:,d);     % Estimated value of prediction output (must be thetae (:, k-d))
    ye=yr(k);                   % Estimated value of predicted output can be yr (k)
    
    %Extract identification parameters
    ge=thetae(1:ng+1,k)'; fe=thetae(ng+2:ng+nf+2,k)'; 
    ce=[1 thetae(ng+nf+3:ng+nf+2+nc,k)'];
    if abs(ce(2))>0.9ce(2)=sign(ce(2))*0.9;end;
    if fe (1)<0.1 fe(1)=0.1;end; 			            % Let the lower bound of f0 be 0.1       
    u(k)=(-fe(2:nf+1)*uk(1:nf)+ce*[yr(k+d:-1:k+d-min(d,nc));
    yrk(1:nc-d)]-ge*[y(k);yk(1:na-1)])/fe(1); 		    % Control input
    
    % Update data
    for i=d:-1:2  thetaek(:,i)=thetaek(:,i-1);end;
    thetaek(:,1)=thetae(:,k);
    for i=d+nf:-1:2 uk(i)=uk(i-1);end;
    uk(1)=u(k);
    for i=d+ng:-1:2 yk(i)=yk(i-1);end;
    yk(1)=y(k);
    for i=nc:-1:2 yek(i)=yek(i-1);yrk(i)=yrk(i-1);xik(i)=xik(i-1);end;
    if nc>0 yek(1)=ye;yrk(1)=yr(k);xik(1)=xi(k);end;
end;

figure(1);
subplot(2,1,1); plot(time,yr(1:L),'r-', time,y,'b-');
xlabel('k'); ylabel('y_r(k), y(k)');
legend('y_r(k)','y(k)'); axis([0 L -20 20]);
subplot(2,1,2);plot(time,u);
xlabel('k'); ylabel('u(k)'); axis([0 L -40 40]);

figure(2);
subplot(211); plot([1:L],thetae(1:ng+1,:),[1:L],thetae(ng+nf+3:ng+2+nf+nc,:));xlabel('k'); 
ylabel('Parameter estimation g, c');
legend('g_0','g_1','c_1'); axis([0 L -3 4]);
subplot(212); plot([1:L],thetae(ng+2:ng+2+nf,:));
xlabel('k'); ylabel('Parameter estimation f');
legend('f_0','f_1','f_2','f_3','f_4'); axis([0 L 0 4]);

