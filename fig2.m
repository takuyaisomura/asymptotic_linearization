
%--------------------------------------------------------------------------------

% fig2.m
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
% initialization

clear
T            = 40000;           % training sample size
NT           = 20;              % number of input dimensionalities

seed         = 0;
rng(1000000+seed);              % set seed for reproducibility

eig_ratio       = cell(2,1);    % eigenvalue ratio
eig_ratio{1,1}  = zeros(28,NT); % for Ns = 10
eig_ratio{2,1}  = zeros(28,NT); % for Ns = 100
est_err         = cell(2,1);    % estimation error
est_err{1,1}    = zeros(28,NT); % for Ns = 10
est_err{2,1}    = zeros(28,NT); % for Ns = 100
est_err2        = cell(2,1);    % estimation error (theory)
est_err2{1,1}   = zeros(28,NT); % for Ns = 10
est_err2{2,1}   = zeros(28,NT); % for Ns = 100

%--------------------------------------------------------------------------------
% run

for i = 1:28
 if     (i <= 10), Nf = i*10;
 elseif (i <= 19), Nf = (i-9)*100;
 else,             Nf = (i-18)*1000; end
 Nx = Nf;
 fprintf(1,'Nx = %d ', Nx)
 for j = 1:2
  Ns = 10^j;
  if (Ns > Nx), continue, end
  for k = 1:NT
   % generative process
   A = randn(Nf, Ns) / sqrt(Ns);      % higher layer wight matrix
   B = randn(Nx, Nf) / sqrt(Nf);      % lower layer wight matrix
   a = randn(Nf, 1) / sqrt(Ns);       % higher layer offsets
   s = rand(Ns, T)*2*sqrt(3)-sqrt(3); % hidden sources
   f = sign(A * s + a * ones(1,T));   % hidden bases
   x = B * f;                         % sensory inputs
   
   % PCA of input covariance
   Cov_x = cov(x');                   % input covariance
   if (Ns == Nx)
    [P,L]   = eig(Cov_x);             % eigenvalue decomposition
   else
    [P,L]   = eigs(Cov_x,Ns);         % eigenvalue decomposition
   end
   PLP     = P * L * P';              % major components
   H       = ((f-mean(f')'*ones(1,T))*s'/T) * (s*s'/T)^(-1/2); % coefficient matrix
   % this treatment is to ajust generalization error
   BAAB    = B * H * H' * B';         % signal covariance
   BSigB   = Cov_x - BAAB;            % noise covariance
   LMmin   = min(abs(eigs(BAAB,Ns))); % minimum eigenvalue of signal covariance
   [U,S]   = eigs(BAAB,Ns);           % eigenvalue decomposition of signal covariance
   EE      = S^-1*U'*BSigB*(eye(Nx)-U*U')*BSigB*U*S^-1;  % theoretical value of estimation error
   eig_ratio{j,1}(i,k)  = abs(eigs(BSigB,1)) / LMmin;    % eigenvalue ratio
   est_err{j,1}(i,k)    = 1 - trace(P'*U*U'*P)/Ns;       % estimation error
   est_err2{j,1}(i,k)   = trace(EE)/Ns;                  % estimation error (theory)
   fprintf(1,'.')
  end
 end
 fprintf(1,' %f, %f, %f, %f, %f, %f\n', mean(eig_ratio{1,1}(i,:)), mean(eig_ratio{2,1}(i,:)), mean(est_err{1,1}(i,:)), mean(est_err{2,1}(i,:)), mean(est_err2{1,1}(i,:)), mean(est_err2{2,1}(i,:)))
 % output file
 csvwrite('eig_ratio_Ns10.csv',[1:NT; eig_ratio{1,1}])
 csvwrite('eig_ratio_Ns100.csv',[1:NT; eig_ratio{2,1}])
 csvwrite('est_err_Ns10.csv',[1:NT; est_err{1,1}])
 csvwrite('est_err_Ns100.csv',[1:NT; est_err{2,1}])
 csvwrite('est_err2_Ns10.csv',[1:NT; est_err2{1,1}])
 csvwrite('est_err2_Ns100.csv',[1:NT; est_err2{2,1}])
end

%--------------------------------------------------------------------------------

