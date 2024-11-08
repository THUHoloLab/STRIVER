% ========================================================================
% Introduction
% ========================================================================
% This code provides a simple demonstration of dynamic phase retrieval via
% spatiotemporal total variation regularization.
%
% Reference:
%   - Y. Gao and L. Cao, "Motion-resolved, reference-free holographic
%     imaging via spatiotemporally regularized inversion," Optica 11(1), 
%     32-41 (2024).
% 
% Author: Yunhui Gao (gyh21@mails.tsinghua.edu.cn)
% =========================================================================
%%
% =========================================================================
% Data generation
% =========================================================================
clear;clc
close all

rng(0);         % set random seed for reproducibility

% load functions
addpath(genpath('./utils'))
addpath(genpath('../src'))

% simulation settings
K = 10;                         % number of measurements
M1 = 256; M2 = 256;             % measurement dimensions

% system parameters
params.pxsize = 2.740e-3;       % sensor pixel size (mm)
params.wavlen = 0.532e-3;       % wavelength (mm)
params.dist_1 = 2;              % sample-to-diffuser distance (mm)
params.dist_2 = 2;              % diffuser-to-sensor distance (mm)

% zero-pad the object to avoid convolution artifacts
padpixels_1 = 20;      % number of padding pixels (for sample-to-diffuser diffraction)
padpixels_2 = 20;      % number of padding pixels (for diffuser-to-sensor diffraction)

% diffuser parameters
mask_feature_size = 4;
mask = imresize(rand(floor((M1+2*padpixels_2)/mask_feature_size),floor((M2+2*padpixels_2)/mask_feature_size)),[M1+2*padpixels_2,M2+2*padpixels_2],'nearest');
mask(mask <  0.5) = 0;
mask(mask >= 0.5) = 1;

params.shift = [linspace(-mask_feature_size*params.pxsize*K/2,mask_feature_size*params.pxsize*K/2,K);linspace(0,0,K)];
shift_range = ceil(max(params.shift(:))/params.pxsize);

% simulate a dynamic sample from the modified Shepp-Logan phantom
img = im2double(imresize(phantom('Modified Shepp-Logan',max([M1,M2])),[M1,M2]));
t_dir = [1,0];
t_dir = t_dir./norm(t_dir,2);
speed_t = 1;    % sample translation speed
speed_r = 1;    % sample rotation speed
N1 = M1 + 2*padpixels_1 + 2*padpixels_2 + 2*shift_range;
N2 = M2 + 2*padpixels_1 + 2*padpixels_2 + 2*shift_range;
vid = nan(N1,N2,K);
for k = 1:K
    vid(:,:,k) = abs(imrotate(imshift(zeropad(img,padpixels_1+padpixels_2+shift_range),speed_t*(k-1)*t_dir(1),speed_t*(k-1)*t_dir(2)),-speed_r*(k-1)+5,'bicubic','crop'));
end
vid = (vid - min(vid(:)))./(max(vid(:)) - min(vid(:)));
x = (1-0.5*vid).*exp(1i*(pi/2*vid));    % sample transmission function

% forward model
T   = @(x,k) imshift(x, params.shift(1,k)/params.pxsize, params.shift(2,k)/params.pxsize);  % lateral displacement operator
TH  = @(x,k) imshift(x,-params.shift(1,k)/params.pxsize,-params.shift(2,k)/params.pxsize);  % Hermitian operator of T
C0  = @(x)   imgcrop(x,shift_range);                                    % image cropping operator
C0T = @(x)   zeropad(x,shift_range);                                    % transpose operator of C0
Q1  = @(x)   propagate(x, params.dist_1,params.pxsize,params.wavlen);   % free-space propagation operator from sample to diffuser
Q1H = @(x)   propagate(x,-params.dist_1,params.pxsize,params.wavlen);   % Hermitian operator of Q1
C1  = @(x)   imgcrop(x,padpixels_1);                                    % image cropping operator
C1T = @(x)   zeropad(x,padpixels_1);                                    % transpose operator of C1
M   = @(x)   x.*mask;                                                   % diffuser modulation operator
MH  = @(x)   x.*conj(mask);                                             % Hermitian operator of M
Q2  = @(x)   propagate(x, params.dist_2,params.pxsize,params.wavlen);   % free-space propagation operator from diffuser to sensor
Q2H = @(x)   propagate(x,-params.dist_2,params.pxsize,params.wavlen);   % Hermitian operator of Q2
C2  = @(x)   imgcrop(x,padpixels_2);                                    % image cropping operator
C2T = @(x)   zeropad(x,padpixels_2);                                    % transpose operator of C2
A   = @(x,k) C2(Q2(M(C1(Q1(C0(T(x,k)))))));                             % overall measurement operator
AH  = @(x,k) TH(C0T(Q1H(C1T(MH(Q2H(C2T(x)))))),k);                      % Hermitian operator of A

% generate data
rng(0)           % random seed, for reproducibility
snr_val = 30;    % noise level
y = zeros(M1,M2,K);
for k = 1:K
    y(:,:,k) = max(awgn(abs(A(x(:,:,k),k)).^2,snr_val),0);
end

% display measurement
figure
set(gcf,'unit','normalized','position',[0.1,0.3,0.8,0.4])
for k = 1:K
    subplot(1,3,1),imshow(abs(x(:,:,k)),[0,1.2]);colorbar
    title('Sample amplitude','interpreter','latex','fontsize',12)
    ax = subplot(1,3,2);imshow(angle(x(:,:,k)),[-pi,pi]);colorbar
    colormap(ax,'inferno')
    title('Sample phase','interpreter','latex','fontsize',12)
    subplot(1,3,3),imshow(y(:,:,k),[]);colorbar
    title('Intensity measurement','interpreter','latex','fontsize',12)
    drawnow;
end

%%
% =========================================================================
% Phase retrieval algorithm
% =========================================================================

K_recon = K;        % number of reconstructed frames

% algorithm settings
x_init = ones(size(x,1),size(x,2),K_recon);     % initial guess
lams_s = [1e-2, 1e-3];                          % regularization parameter (spatial)
lams_t = [1e-2, 1e-3];                          % regularization parameter (temporal)

alph = 10;              % hyperparameter for tuning regularization weights
gam  = 2;               % step size (see the paper for details)
n_iters    = 200;       % number of iterations (main loop)
n_subiters = 1;         % number of subiterations (proximal update)

% options
opts.verbose = true;        % display status during the iterations
opts.errfunc = [];          % user-defined error metrics
opts.lams = @(iter) reg_param(iter, n_iters/2, alph, [lams_s(1),lams_s(1),lams_t(1)], [lams_s(2),lams_s(2),lams_t(2)]);
opts.display = true;        % display intermediate results during the iterations

% function handles to calculate objective function and gradients
myF     = @(x) F(x,y,A,K,K_recon);              % data-fidelity function 
mydF    = @(x) dF(x,y,A,AH,K,K_recon);          % gradient of the data-fidelity function
myR     = @(x) normTV(x);                       % regularization function
myproxR = @(x,gam) proxTV(x,gam,n_subiters);    % proximal operator for the regularization function

% run the proximal gradient algorithm
[x_est,J_vals,runtimes] = APG(x_init,myF,mydF,myR,myproxR,gam,n_iters,opts);

%%
% =========================================================================
% Display results
% =========================================================================
addpath(genpath('utils/cmap-master'))
figure
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])
for k = 1:K
    subplot(1,2,1),imshow(abs(C2(C1(C0(x_est(:,:,k))))),[0,1.2]);colorbar
    title('Retrieved sample amplitude','interpreter','latex','fontsize',12)
    ax = subplot(1,2,2);imshow(angle(C2(C1(C0(x_est(:,:,ceil(K_recon*k/K)))))),[-pi,pi]);colorbar
    colormap(ax,'inferno')
    title('Retrieved sample phase','interpreter','latex','fontsize',12)
    drawnow;
end

%%
% =========================================================================
% Calculate errors
% =========================================================================
res = 0; ref = 0;
KK   = @(k) ceil(K_recon*k/K);
for k = 1:K
    amp_gt  = abs(C2(C1(C0(x(:,:,k)))));
    pha_gt  = angle(C2(C1(C0(x(:,:,k) * exp(1i*-pi/4)))));
    amp_est = abs(C2(C1(C0(x_est(:,:,KK(k))))));
    pha_est = angle(C2(C1(C0(x_est(:,:,KK(k)) * exp(1i*-pi/4)))));
    pha_est = pha_est - mean(pha_est(:)) + mean(pha_gt(:));
    x_est_tmp = amp_est.*exp(1i*pha_est);
    x_tmp = amp_gt.*exp(1i*pha_gt);
    res = res + norm(x_est_tmp(:) - x_tmp(:),2).^2;
    ref = ref + norm(x_tmp(:),2).^2;
end
err = sqrt(res/ref);
disp(['Relative error: ',num2str(err)])

%%
% =========================================================================
% Auxiliary functions
% =========================================================================

function v = F(x,y,A,K,K_recon)
% =========================================================================
% Data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity image.
%           - A   : The sampling operator.
%           - K   : Number of diversity measurements.
% Output:   - v   : Value of the fidelity function.
% =========================================================================
v = 0;
for k = 1:K
    v = v + 1/2/K * norm2(abs(A(x(:,:,KK(k)),k)) - sqrt(y(:,:,k)))^2;
end

function n = norm2(x)   % calculate the l2 vector norm
n = norm(x(:),2);
end

function index = KK(k)
index = ceil(K_recon*k/K);
end

end


function g = dF(x,y,A,AH,K,K_recon)
% =========================================================================
% Gradient of the data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity image.
%           - A   : The sampling operator.
%           - AH  : Hermitian of A.
%           - K   : Number of diversity measurements.
% Output:   - g   : Wirtinger gradient.
% =========================================================================
g = zeros(size(x));
for k = 1:K
    u = A(x(:,:,KK(k)),k);
    u = u.*(1 - sqrt(y(:,:,k))./(abs(u)));
    g(:,:,KK(k)) = g(:,:,KK(k)) + 1/2/(K/K_recon) * AH(u,k);
end

function index = KK(k)
index = ceil(K_recon*k/K);
end

end


function u = imgcrop(x,cropsize)
% =========================================================================
% Crop the central part of the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - cropsize : Cropping pixel number along each dimension.
% Output:   - u        : Cropped image.
% =========================================================================
u = x(cropsize+1:end-cropsize,cropsize+1:end-cropsize);
end


function u = zeropad(x,padsize)
% =========================================================================
% Zero-pad the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - padsize  : Padding pixel number along each dimension.
% Output:   - u        : Zero-padded image.
% =========================================================================
u = padarray(x,[padsize,padsize],0);
end


function [lamval_1, lamval_2, lamval_3] = reg_param(iter,n_iters,alph,lams_start,lams_end)
% =========================================================================
% Setting the regularization parameter tau during the iterations using an
% sigmoid function.
% -------------------------------------------------------------------------
% Input:    - iter     : The current iteration number.
%           - n_iters  : Total iteration numbers.
%           - alph     : Sigmoid function parameter.
%           - tau1     : Initial regularization parameter.
%           - tau2     : Final regularization parameter.
% Output:   - lam_val  : The current regularization parameter.
% =========================================================================
global lam_1; global lam_2; global lam_3;
lam_1 = (lams_start(1)-lams_end(1))*1./(1+exp(alph*(iter/n_iters - 1/2))) + lams_end(1);
lam_2 = (lams_start(2)-lams_end(2))*1./(1+exp(alph*(iter/n_iters - 1/2))) + lams_end(2);
lam_3 = (lams_start(3)-lams_end(3))*1./(1+exp(alph*(iter/n_iters - 1/2))) + lams_end(3);
lamval_1 = lam_1; lamval_2 = lam_2; lamval_3 = lam_3;
end
