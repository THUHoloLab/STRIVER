% ========================================================================
% Introduction
% ========================================================================
% This code provides a simple demonstration of dynamic phase retrieval via
% spatiotemporal total variation regularization.
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
K = 10;
Mx = 256; My = 256;
img = im2double(imresize(phantom('Modified Shepp-Logan',max([Mx,My])),[Mx,My]));

% system parameters
params.pxsize = 2.740e-3;       % sensor pixel size (mm)
params.wavlen = 0.532e-3;       % wavelength (mm)
params.dist_1 = 2;              % sample-to-diffuser distance (mm)
params.dist_2 = 2;              % diffuser-to-sensor distance (mm)

% zero-pad the object to avoid convolution artifacts
padpixels_1 = 20;      % number of padding pixels (for sample-to-diffuser diffraction)
padpixels_2 = 20;      % number of padding pixels (for diffuser-to-sensor diffraction)

% physical parameters
mask_feature_size = 2;
mask = imresize(rand(floor((Mx+2*padpixels_2)/mask_feature_size),floor((My+2*padpixels_2)/mask_feature_size)),[Mx+2*padpixels_2,My+2*padpixels_2],'nearest');
mask(mask < 0.5)  = 0;
mask(mask >= 0.5) = 1;

params.shift = [linspace(0,0,K);linspace(-mask_feature_size*params.pxsize*K/2,mask_feature_size*params.pxsize*K/2,K)];
shift_range = ceil(max(params.shift(:))/params.pxsize);


speed_t = 1;    % sample translation speed
speed_r = 1;    % sample rotation speed

Nx = Mx + 2*padpixels_1 + 2*padpixels_2 + 2*shift_range;
Ny = My + 2*padpixels_1 + 2*padpixels_2 + 2*shift_range;
vid = nan(Nx,Ny,K);
for k = 1:K
    vid(:,:,k) = abs(imrotate(imshift(zeropad(img,padpixels_1+padpixels_2+shift_range),speed_t*(k-1),0),-speed_r*(k-1),'bicubic','crop'));
end
vid = (vid - min(vid(:)))./(max(vid(:)) - min(vid(:)));
% sample
x = (1-0.5*vid).*exp(1i*(pi*vid - pi/2));


% forward model
T   = @(x,k) imshift(x, params.shift(1,k)/params.pxsize, params.shift(2,k)/params.pxsize);
TH  = @(x,k) imshift(x,-params.shift(1,k)/params.pxsize,-params.shift(2,k)/params.pxsize);
C0  = @(x)   imgcrop(x,shift_range);
C0T = @(x)   zeropad(x,shift_range);
Q1  = @(x)   propagate(x, params.dist_1,params.pxsize,params.wavlen);
Q1H = @(x)   propagate(x,-params.dist_1,params.pxsize,params.wavlen);
C1  = @(x)   imgcrop(x,padpixels_1);
C1T = @(x)   zeropad(x,padpixels_1);
M   = @(x)   x.*mask;
MH  = @(x)   x.*conj(mask);
Q2  = @(x)   propagate(x, params.dist_2,params.pxsize,params.wavlen);
Q2H = @(x)   propagate(x,-params.dist_2,params.pxsize,params.wavlen);
C2  = @(x)   imgcrop(x,padpixels_2);
C2T = @(x)   zeropad(x,padpixels_2);
A   = @(x,k) C2(Q2(M(C1(Q1(C0(T(x,k)))))));
AH  = @(x,k) TH(C0T(Q1H(C1T(MH(Q2H(C2T(x)))))),k);

% generate data
rng(0)           % random seed, for reproducibility
snr_val = 40;    % noise level
y = zeros(Mx,My,K);
for k = 1:K
    y(:,:,k) = max(awgn(abs(A(x(:,:,k),k)).^2,snr_val),0);
end

% display measurement
figure
set(gcf,'unit','normalized','position',[0.1,0.3,0.8,0.4])
for k = 1:K
    subplot(1,3,1),imshow(abs(x(:,:,k)),[0,1]);colorbar
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

clear functions     % release memory (if using puma)

K_recon = K;

% algorithm settings
x_init = ones(size(x,1),size(x,2),K_recon);     % initial guess
lams_s = [1e-2, 1e-4];                          % regularization parameter (spatial)
lams_t = [1e-2, 2e-4];                        % regularization parameter (temporal)

alpha = 10;
gamma = 2;                  % step size (see the paper for details)
n_iters = 200;              % number of iterations (main loop)
n_subiters = 1;             % number of iterations (denoising)

% options
opts.verbose = true;        % display status during the iterations
opts.errfunc = [];          % user-defined error metrics
opts.lams = @(iter) reg_param(iter, n_iters/2, alpha, [lams_s(1),lams_s(1),lams_t(1)], [lams_s(2),lams_s(2),lams_t(2)]);


% function handles
myF     = @(x) F(x,y,A,K,K_recon);
mydF    = @(x) dF(x,y,A,AH,K,K_recon);
myR     = @(x) normTV(x);
myproxR = @(x,gam) proxTV(x,gam,n_subiters);

[x_est,J_vals,runtimes] = fista_visualize(x_init,myF,mydF,myR,myproxR,gamma,n_iters,opts);



%%
% =========================================================================
% Display results
% =========================================================================
addpath(genpath('utils/cmap-master'))
figure
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])
for k = 1:K
    subplot(1,2,1),imshow(abs(C2(C1(C0(x_est(:,:,k))))),[0,1]);colorbar
    title('Retrieved sample amplitude','interpreter','latex','fontsize',12)
    ax = subplot(1,2,2);imshow(angle(C2(C1(C0(x_est(:,:,ceil(K_recon*k/K)))))),[-pi,pi]);colorbar
    colormap(ax,'inferno')
    title('Retrieved sample phase','interpreter','latex','fontsize',12)
    drawnow;
end

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

