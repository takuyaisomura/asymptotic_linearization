
%--------------------------------------------------------------------------------

% fig3.m
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
NT           = 20;              % number of trials
eta          = 0.025;           % learning rate for ICA
Num          = 5000;            % number of iterations for ICA

seed         = 0;
rng(1000000+seed);              % set seed for reproducibility

%--------------------------------------------------------------------------------
% fig 3a

fprintf(1, 'fig 3a\n')
errors = zeros(5,2);
Ns = 100;
for i = 1:3
 if (i == 1), Nf = 500; end
 if (i == 2), Nf = 1000; end
 if (i == 3), Nf = 10000; end
 Nx = Nf;
 fprintf(1,'generative process\n');
 A = randn(Nf, Ns) / sqrt(Ns);      % higher layer wight matrix
 B = randn(Nx, Nf) / sqrt(Nf);      % lower layer wight matrix
 a = randn(Nf, 1) / sqrt(Ns);       % higher layer offsets
 s = rand(Ns, T)*2*sqrt(3)-sqrt(3); % hidden sources
 f = sign(A * s + a * ones(1,T));   % hidden bases
 x = B * f;                         % sensory inputs
 
 fprintf(1,'PCA\n');
 Cov_x = cov(x');
 if (Ns == Nx)
  [P,L]   = eig(Cov_x);
 else
  [P,L]   = eigs(Cov_x,Ns);
 end
 W = L^(-1/2) * P';
 u = W * (x - mean(x')'*ones(1,T));
 
 fprintf(1,'ICA ');
 [v,Wica] = ica(u,s,eta,Num);
 fprintf(1,'%f\n', sum(mean((v-s)'.^2))/Ns);
 subplot(1,2,1), image(abs(Wica*W*B*A)*100)
 subplot(1,2,2), image(abs(corr(v',s'))*100)
 drawnow
 if (i == 1), csvwrite('s_v_Ns100Nx500.csv',[1:Ns 1:Ns; s(:,1:T/10)' v(:,1:T/10)']), end
 if (i == 2), csvwrite('s_v_Ns100Nx1000.csv',[1:Ns 1:Ns; s(:,1:T/10)' v(:,1:T/10)']), end
 if (i == 3), csvwrite('s_v_Ns100Nx10000.csv',[1:Ns 1:Ns; s(:,1:T/10)' v(:,1:T/10)']), end
 if (i == 1), csvwrite('cov_Ns100Nx500.csv',[1:Ns; v*s'/T]), end
 if (i == 2), csvwrite('cov_Ns100Nx1000.csv',[1:Ns; v*s'/T]), end
 if (i == 3), csvwrite('cov_Ns100Nx10000.csv',[1:Ns; v*s'/T]), end
 
 BH      = (x-mean(x')'*ones(1,T))*s'/T;
 BSigmaB = Cov_x - BH*BH';
 BHinv   = (BH'*BH)^(-1)*BH';
 errors(i,1) = sum(mean((v-s)'.^2))/Ns;
 errors(i,2) = trace(BHinv*BSigmaB*BHinv')/Ns;
end

%--------------------------------------------------------------------------------
% fig 3b

fprintf(1, 'fig 3b\n')
Ns = 100;
Nf = 10000;
Nx = Nf;
fprintf(1,'generative process\n');
A = randn(Nf, Ns) / sqrt(Ns);      % higher layer wight matrix
B = randn(Nx, Nf) / sqrt(Nf);      % lower layer wight matrix
a = randn(Nf, 1) / sqrt(Ns);       % higher layer offsets
s = rand(Ns, T)*2*sqrt(3)-sqrt(3); % hidden sources
f = (A * s + a * ones(1,T)).^3;    % hidden bases
x = B * f;                         % sensory inputs

fprintf(1,'PCA\n');
Cov_x = cov(x');
if (Ns == Nx)
 [P,L]   = eig(Cov_x);
else
 [P,L]   = eigs(Cov_x,Ns);
end
W = L^(-1/2) * P';
u = W * (x - mean(x')'*ones(1,T));

fprintf(1,'ICA ');
[v,Wica] = ica(u,s,eta,Num);
fprintf(1,'%f\n', sum(mean((v-s)'.^2))/Ns);
subplot(1,2,1), image(abs(Wica*W*B*A)*100)
subplot(1,2,2), image(abs(corr(v',s'))*100)
drawnow
csvwrite('s_v_Ns100Nx10000cubic.csv',[1:Ns 1:Ns; s(:,1:T/10)' v(:,1:T/10)'])
csvwrite('cov_Ns100Nx10000cubic.csv',[1:Ns; v*s'/T])

BH      = (x-mean(x')'*ones(1,T))*s'/T;
BSigmaB = Cov_x - BH*BH';
BHinv   = (BH'*BH)^(-1)*BH';
errors(4,1) = sum(mean((v-s)'.^2))/Ns;
errors(4,2) = trace(BHinv*BSigmaB*BHinv')/Ns;

%--------------------------------------------------------------------------------
% fig 3c

fprintf(1, 'fig 3c\n')
Ns = 100;
Nf = 10000;
Nx = Nf;
fprintf(1,'generative process\n');
A = randn(Nf, Ns) / sqrt(Ns);      % higher layer wight matrix
B = randn(Nx, Nf) / sqrt(Nf);      % lower layer wight matrix
a = randn(Nf, 1) / sqrt(Ns);       % higher layer offsets
s = rand(Ns, T)*2*sqrt(3)-sqrt(3); % hidden sources
f = (A * s + a * ones(1,T)) .* (((A * s + a * ones(1,T))) > 0); % hidden bases
x = B * f;                         % sensory inputs

fprintf(1,'PCA\n');
Cov_x = cov(x');
if (Ns == Nx)
 [P,L]   = eig(Cov_x);
else
 [P,L]   = eigs(Cov_x,Ns);
end
W = L^(-1/2) * P';
u = W * (x - mean(x')'*ones(1,T));

fprintf(1,'ICA ');
[v,Wica] = ica(u,s,eta,Num);
fprintf(1,'%f\n', sum(mean((v-s)'.^2))/Ns);
subplot(1,2,1), image(abs(Wica*W*B*A)*100)
subplot(1,2,2), image(abs(corr(v',s'))*100)
drawnow
csvwrite('s_v_Ns100Nx10000relu.csv',[1:Ns 1:Ns; s(:,1:T/10)' v(:,1:T/10)'])
csvwrite('cov_Ns100Nx10000relu.csv',[1:Ns; v*s'/T])

BH      = (x-mean(x')'*ones(1,T))*s'/T;
BSigmaB = Cov_x - BH*BH';
BHinv   = (BH'*BH)^(-1)*BH';
errors(5,1) = sum(mean((v-s)'.^2))/Ns;
errors(5,2) = trace(BHinv*BSigmaB*BHinv')/Ns;
csvwrite('bss_errors_fig3abc.csv',[1:2; errors])

%--------------------------------------------------------------------------------
% fig 3d

fprintf(1, 'fig 3d\n')
bss_err       = cell(2,1);
bss_err{1,1}  = zeros(30,NT);
bss_err{2,1}  = zeros(30,NT);
bss_err2      = cell(2,1);
bss_err2{1,1} = zeros(30,NT);
bss_err2{2,1} = zeros(30,NT);

for i = 1:30
 if     (i <= 10), Nf = i*10;
 elseif (i <= 19), Nf = (i-9)*100;
 elseif (i <= 28), Nf = (i-18)*1000;
 else,             Nf = (i-27)*10000; end
 Nx = Nf;
 fprintf(1,'Nx = %d\n', Nx)
 for j = 1:2
  Ns = 10^j;
  if (Ns > Nx), continue, end
  for k = 1:NT
   fprintf(1,'generative process\n');
   A = randn(Nf, Ns) / sqrt(Ns);      % higher layer wight matrix
   B = randn(Nx, Nf) / sqrt(Nf);      % lower layer wight matrix
   a = randn(Nf, 1) / sqrt(Ns);       % higher layer offsets
   s = rand(Ns, T)*2*sqrt(3)-sqrt(3); % hidden sources
   f = sign(A * s + a * ones(1,T));   % hidden bases
   x = B * f;                         % sensory inputs
   
   fprintf(1,'PCA\n');
   Cov_x = cov(x');
   if (Ns == Nx)
    [P,L]   = eig(Cov_x);
   else
    [P,L]   = eigs(Cov_x,Ns);
   end
   W = L^(-1/2) * P';
   u = W * (x - mean(x')'*ones(1,T));
   
   fprintf(1,'ICA ');
   [v,Wica] = ica(u,s,eta,Num);
   bss_err{j,1}(i,k) = sum(mean((v-s)'.^2))/Ns;
   
   BH      = (x-mean(x')'*ones(1,T))*s'/T;
   BSigmaB = Cov_x - BH*BH';
   BHinv   = (BH'*BH)^(-1)*BH';
   bss_err2{j,1}(i,k) = trace(BHinv*BSigmaB*BHinv')/Ns;
   
   fprintf(1,'%d, %d, %f, %f, %f\n', i, k, bss_err{j,1}(i,k),bss_err2{j,1}(i,k),(1/(2/pi)-1)*Ns/Nx*2+1/2/Ns);
   subplot(1,2,1), image(abs(Wica*W*B*A)*100)
   subplot(1,2,2), image(abs(corr(v',s'))*100)
   drawnow
  end
  if (j == 1), csvwrite('bss_err_Ns10.csv',[1:NT; bss_err{1,1}]), end
  if (j == 2), csvwrite('bss_err_Ns100.csv',[1:NT; bss_err{2,1}]), end
  if (j == 1), csvwrite('bss_err2_Ns10.csv',[1:NT; bss_err2{1,1}]), end
  if (j == 2), csvwrite('bss_err2_Ns100.csv',[1:NT; bss_err2{2,1}]), end
 end
end

%--------------------------------------------------------------------------------
% fig 3d with truncated normal distribution

fprintf(1, 'fig 3d with truncated normal distribution\n')
bss_err       = cell(2,1);
bss_err{1,1}  = zeros(30,NT);
bss_err{2,1}  = zeros(30,NT);
bss_err2      = cell(2,1);
bss_err2{1,1} = zeros(30,NT);
bss_err2{2,1} = zeros(30,NT);

for i = 28:30
 if     (i <= 10), Nf = i*10;
 elseif (i <= 19), Nf = (i-9)*100;
 elseif (i <= 28), Nf = (i-18)*1000;
 else,             Nf = (i-27)*10000; end
 Nx = Nf;
 fprintf(1,'Nx = %d\n', Nx)
 for j = 1:1
  Ns = 10^j;
  if (Ns > Nx), continue, end
  for k = 1:NT
   fprintf(1,'generative process\n');
   A = randn(Nf, Ns) / sqrt(Ns);      % higher layer wight matrix
   B = randn(Nx, Nf) / sqrt(Nf);      % lower layer wight matrix
   a = randn(Nf, 1) / sqrt(Ns);       % higher layer offsets
   s = zeros(Ns, T);
   % symmetric truncated normal distribution
   for l = 1:Ns
    temp   = randn(1,T*2);
    temp   = temp((-2.55 < temp) & (temp < 2.55));
    s(l,:) = temp(1,1:T);             % hidden sources
   end
   s = diag(std(s'))^-1 * s;
   fprintf(1,'kurtosis = %f\n', mean(kurtosis(s')));
   f = sign(A * s + a * ones(1,T));   % hidden bases
   x = B * f;                         % sensory inputs
   
   fprintf(1,'PCA\n');
   Cov_x = cov(x');
   if (Ns == Nx)
    [P,L]   = eig(Cov_x);
   else
    [P,L]   = eigs(Cov_x,Ns);
   end
   W = L^(-1/2) * P';
   u = W * (x - mean(x')'*ones(1,T));
   
   fprintf(1,'ICA ');
   [v,Wica] = ica(u,s,eta,Num*2);
   bss_err{j,1}(i,k) = sum(mean((v-s)'.^2))/Ns;
   
   BH      = (x-mean(x')'*ones(1,T))*s'/T;
   BSigmaB = Cov_x - BH*BH';
   BHinv   = (BH'*BH)^(-1)*BH';
   bss_err2{j,1}(i,k) = trace(BHinv*BSigmaB*BHinv')/Ns;
   
   fprintf(1,'%d, %d, %f, %f, %f\n', i, k, bss_err{j,1}(i,k),bss_err2{j,1}(i,k),(1/(2/pi)-1)*Ns/Nx*2+1/2/Ns);
   subplot(1,2,1), image(abs(Wica*W*B*A)*100)
   subplot(1,2,2), image(abs(corr(v',s'))*100)
   drawnow
  end
  if (j == 1), csvwrite('bss_err_Ns10truncated.csv',[1:NT; bss_err{1,1}]), end
  if (j == 1), csvwrite('bss_err2_Ns10truncated.csv',[1:NT; bss_err2{1,1}]), end
 end
end

%--------------------------------------------------------------------------------

