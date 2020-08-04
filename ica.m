
%--------------------------------------------------------------------------------

% ica.m
%
% This demo is included in
% On the achievability of blind source separation for high-dimensional nonlinear source mixtures
% Takuya Isomura, Taro Toyoizumi
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/asymptotic_linearization
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-8-3
%

%--------------------------------------------------------------------------------

function [v,Wica] = ica(u,s,eta,rep)
Ns   = length(u(:,1));
T    = length(u(1,:));
Wica = diag(std(u'))^(-1);
eta2 = eta*2;
for t = 1:rep
 if (t == rep/2), eta2 = eta; end
 % neural activity
 rnd = randi([1 T],1,T/10);
 v = Wica * u(:,rnd);
 % Amari's ICA rule
 Wica = Wica + eta2 * (eye(Ns) - (v.^3/3)*v'/(T/10)) * Wica;
 if (rem(t,rep/10) == 0), fprintf(1,'.'), end
end
fprintf(1,'\n')

v = Wica * u;
Omega = corr(v',s');
for row = 1:Ns
 col = find(abs(Omega(row,:)) == max(abs(Omega(row,:))));
 sgn = Omega(row,col);
 Omega(:,col) = 0;
 Omega(row,col) = sgn;
end
Wica2 = Omega' * Wica;
v = Wica2 * u;
Wica = diag(std(v'))^(-1) * Wica2;
v = Wica * u;

%--------------------------------------------------------------------------------

