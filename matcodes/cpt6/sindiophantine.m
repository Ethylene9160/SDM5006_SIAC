function [e,f,g]=sindiophantine(a,b,c,d)
% Function: single-step solution of Diophanine equation
% Call format: [e, f, g]=sindiophantine (a, b, c, d)
% Input parameters: polynomial A, B, C coefficients (row vector) and pure lag (4 in total)
% Output parameters: solution e, f, g of Diophanine equation (3 in total)

na=length(a)-1; nb=length(b)-1; nc=length(c)-1;         % The orders of A, B and C
ne=d-1; ng=na-1;                                        % The orders of E and G
ad=[a,zeros(1,ng+ne+1-na)]; cd=[c, zeros(1,ng+d-nc)];  	% Let a (na+2)=a (na+3)== 0
e(1)=1;
for i=2:ne+1
    e(i)=0;
    for j=2:i  e(i)=e(i)+e(i+1-j)*ad(j); end;
    e(i)=cd(i)-e(i);                            % Calculate ei
end;
for i=1:ng+1
     g(i)=0;
     for j=1:ne+1   g(i)=g(i)+e(ne+2-j)*ad(i+j);  end
     g(i)=cd(i+d)-g(i);                         % Calculate gi
end;
f=conv(b,e);                                    % Calculate F
