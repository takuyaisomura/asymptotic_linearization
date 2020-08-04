
%--------------------------------------------------------------------------------

% fig4.m
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
T            = 40000;         % training sample size
Num          = 5000;          % number of iterations for PCA
Num2         = 10000;         % number of iterations for ICA
NT           = 20;            % number of trials

seed         = 0;
rng(1000000+seed);            % set seed for reproducibility

est_err      = cell(2,1);     % estimation error
est_err{1,1} = zeros(101,NT); % for Nx = 1000
est_err{2,1} = zeros(101,NT); % for Nx = 10000
bss_err      = cell(2,1);     % BSS error
bss_err{1,1} = zeros(101,NT); % for Nx = 1000
bss_err{2,1} = zeros(101,NT); % for Nx = 10000

Nxlist = [1000 10000];        % input dimensionality
Nslist = [100 100];           % source dimensionality
eta1   = 0.001;               % learning rate for PCA
eta2   = 0.02;                % learning rate for ICA

%--------------------------------------------------------------------------------
% run

for i = 1:2
 for k = 1:NT
  fprintf(1,'generative process\n');
  Nx = Nxlist(i);
  Nf = Nx;
  Ns = Nslist(i);
  % generative process
  A = randn(Nf, Ns) / sqrt(Ns);      % higher layer wight matrix
  B = randn(Nx, Nf) / sqrt(Nf);      % lower layer wight matrix
  a = randn(Nf, 1) / sqrt(Ns);       % higher layer offsets
  Wpca = randn(Ns, Nx) / sqrt(Nx);   % synaptic weight matrix for PCA
  s = rand(Ns, T)*2*sqrt(3)-sqrt(3); % hidden sources
  f = sign(A * s + a * ones(1,T));   % hidden bases
  x = B * f;                         % sensory inputs
  x = x - mean(x')'*ones(1,T);
  H = (f-mean(f')'*ones(1,T))*s'/T;  % coefficient matrix
  [U,S] = eigs(B*H*H'*B',Ns);        % eigenvalue decomposition of signal covariance
  
  fprintf(1,'PCA\n');
  XX = cov(x');                      % input covariance
  est_err{i,1}(1,k) = 1-trace(Wpca*U*U'*Wpca')/Ns;
  for t = 1:Num
   UX   = Wpca*XX;
   % Oja's subspace rule for PCA
   Wpca = Wpca + eta1 * (UX - UX*Wpca'*Wpca);
   if (rem(t,1000)==0)
    fprintf(1,'%d, %d, %d, %f\n', i, k, t, 1-trace(Wpca*U*U'*Wpca')/Ns);
   end
   if (rem(t,50) == 0), est_err{i,1}(t/50+1,k) = 1-trace(Wpca*U*U'*Wpca')/Ns; end
  end
  
  fprintf(1,'ICA\n');
  u = Wpca * x;                      % encoders
  Wica = diag(std(u'))^(-1);         % synaptic weight matrix for ICA
  v = Wica * u;                      % independent encoders
  Omega = corr(v',s');
  for row = 1:Ns
   col = find(abs(Omega(row,:)) == max(abs(Omega(row,:))));
   sgn = Omega(row,col);
   Omega(:,col) = 0;
   Omega(row,col) = sgn;
  end
  Wica2 = Omega' * Wica;
  v = Wica2 * u;
  v = diag(std(v'))^(-1) * v;
  bss_err{i,1}(1,k) = sum(mean((v-s)'.^2))/Ns;
  
  for t = 1:Num2
   % neural activity
   rnd = randi([1 T],1,T/10);
   v = Wica * u(:,rnd);
   % Amari's ICA rule
   Wica = Wica + eta2 * (eye(Ns) - (v.^3/3)*v'/(T/10)) * Wica;
   if (rem(t,100) == 0)
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
    v = diag(std(v'))^(-1) * v;
    if (rem(t,1000) == 0)
     fprintf(1,'%d, %d, %d, %f\n', i, k, t, sum(mean((v-s)'.^2))/Ns);
     subplot(1,2,1), image(abs(Wica2*Wpca*B*A)*100)
     subplot(1,2,2), image(abs(corr(v',s'))*100)
     drawnow
    end
    bss_err{i,1}(t/100+1,k) = sum(mean((v-s)'.^2))/Ns;
   end
  end
  
  if (i == 1), csvwrite('oja_err_Ns100Nx1000.csv',[1:NT; est_err{1,1}]), end
  if (i == 2), csvwrite('oja_err_Ns100Nx10000.csv',[1:NT; est_err{2,1}]), end
  if (i == 1), csvwrite('bss_err_Ns100Nx1000.csv',[1:NT; bss_err{1,1}]), end
  if (i == 2), csvwrite('bss_err_Ns100Nx10000.csv',[1:NT; bss_err{2,1}]), end
 end
end

%--------------------------------------------------------------------------------

