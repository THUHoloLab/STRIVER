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

% pre-calculate the transfer functions for diffraction modeling
HT  = nan(N1,N2,K);
for k = 1:K
    HT(:,:,k)  = fftshift(transfunc_imshift(N1,N2,params.shift(1,k)/params.pxsize, params.shift(2,k)/params.pxsize));
end
HQ1 = fftshift(transfunc_propagate(N1-2*shift_range,N2-2*shift_range, params.dist_1,params.pxsize,params.wavlen)); % forward propagation transfer function (from sample to diffuser)
HQ2 = fftshift(transfunc_propagate(M1+2*padpixels_2,M2+2*padpixels_2, params.dist_2,params.pxsize,params.wavlen)); % forward propagation transfer function (from diffuser to sensor)

% forward model
T   = @(x,k) ifft2(fft2(x).*HT(:,:,k));             % lateral displacement operator
TH  = @(x,k) ifft2(fft2(x).*conj(HT(:,:,k)));       % Hermitian operator of T
C0  = @(x)   imgcrop(x,shift_range);                % image cropping operator
C0T = @(x)   zeropad(x,shift_range);                % transpose operator of C0
Q1  = @(x)   ifft2(fft2(x).*HQ1);                   % free-space propagation operator from sample to diffuser
Q1H = @(x)   ifft2(fft2(x).*conj(HQ1));             % Hermitian operator of Q1
C1  = @(x)   imgcrop(x,padpixels_1);                % image cropping operator
C1T = @(x)   zeropad(x,padpixels_1);                % transpose operator of C1
M   = @(x)   x.*mask;                               % diffuser modulation operator
MH  = @(x)   x.*conj(mask);                         % Hermitian operator of M
Q2  = @(x)   ifft2(fft2(x).*HQ2);                   % free-space propagation operator from diffuser to sensor
Q2H = @(x)   ifft2(fft2(x).*conj(HQ2));             % Hermitian operator of Q2
C2  = @(x)   imgcrop(x,padpixels_2);                % image cropping operator
C2T = @(x)   zeropad(x,padpixels_2);                % transpose operator of C2
A   = @(x,k) C2(Q2(M(C1(Q1(C0(T(x,k)))))));         % overall measurement operator
AH  = @(x,k) TH(C0T(Q1H(C1T(MH(Q2H(C2T(x)))))),k);  % Hermitian operator of A

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

gpu = false;        % whether using GPU or not

K_recon = K;        % number of reconstructed frames

% algorithm settings
x_est = ones(size(x,1),size(x,2),K_recon);      % initial guess
lams_s = [1e-2, 1e-3];                          % regularization parameter (spatial)
lams_t = [1e-2, 1e-3];                          % regularization parameter (temporal)

alph = 10;              % hyperparameter for tuning regularization weights
gam  = 2;               % step size (see the paper for details)
n_iters    = 200;       % number of iterations (main loop)
n_subiters = 10;        % number of subiterations (proximal update)

% auxilary variables
z_est = x_est;
g_est = zeros(size(x_est));
v_est = zeros(size(x_est,1),size(x_est,2),size(x_est,3),3);
w_est = zeros(size(x_est,1),size(x_est,2),size(x_est,3),3);

% auxilary functions
lams = @(iter) reg_param(iter, n_iters/2, alph, [lams_s(1),lams_s(1),lams_t(1)], [lams_s(2),lams_s(2),lams_t(2)]);
KK   = @(k) ceil(K_recon*k/K);

% initialize GPU
if gpu
    device  = gpuDevice();
    reset(device)
    x_est   = gpuArray(x_est);
    y       = gpuArray(y);
    HT      = gpuArray(HT);
    HQ1     = gpuArray(HQ1);
    HQ2     = gpuArray(HQ2);
    mask    = gpuArray(mask);
    g_est   = gpuArray(g_est);
    z_est   = gpuArray(z_est);
    v_est   = gpuArray(v_est);
    w_est   = gpuArray(w_est);
end

% main loop
timer = tic;
for iter = 1:n_iters

    % print status
    fprintf('iter: %4d / %4d \n', iter, n_iters);
    
    % gradient update
    g_est(:) = 0;
    for k = 1:K
        u = A(z_est(:,:,KK(k)),k);
        u = (abs(u) - sqrt(y(:,:,k))) .* exp(1i*angle(u));
        g_est(:,:,KK(k)) = g_est(:,:,KK(k)) + 1/2/(K/K_recon) * AH(u,k);
    end
    u = z_est - gam * g_est;

    % proximal update
    v_est(:) = 0; w_est(:) = 0;
    [lam_1,lam_2,lam_3] = lams(iter);
    for subiter = 1:n_subiters
        w_next = v_est + 1/12/gam*Df(u-gam*DTf(v_est));
        w_next(:,:,:,1) = min(abs(w_next(:,:,:,1)),lam_1).*exp(1i*angle(w_next(:,:,:,1)));
        w_next(:,:,:,2) = min(abs(w_next(:,:,:,2)),lam_2).*exp(1i*angle(w_next(:,:,:,2)));
        w_next(:,:,:,3) = min(abs(w_next(:,:,:,3)),lam_3).*exp(1i*angle(w_next(:,:,:,3)));

        v_est = w_next + subiter/(subiter+3)*(w_next-w_est);
        w_est = w_next;
    end
    x_next = u - gam*DTf(w_est);
    
    % Nesterov extrapolation
    z_est = x_next + (iter/(iter+3))*(x_next - x_est);
    x_est = x_next;
end

% wait for GPU
if gpu; wait(device); end
toc(timer)

% gather data from GPU
if gpu
    x_est   = gather(x_est);
    y       = gather(y);
    HT      = gather(HT);
    HQ1     = gather(HQ1);
    HQ2     = gather(HQ2);
    mask    = gather(mask);
%     g_est   = gather(g_est);
%     z_est   = gather(z_est);
%     v_est   = gather(v_est);
%     w_est   = gather(w_est);
%     reset(device);
end

%%
% =========================================================================
% Display results
% =========================================================================
addpath(genpath('utils/cmap-master'))
figure
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])
for k = 1:K
    subplot(1,2,1),imshow(abs(C2(C1(C0(x_est(:,:,KK(k)))))),[0,1.2]);colorbar
    title('Retrieved sample amplitude','interpreter','latex','fontsize',12)
    ax = subplot(1,2,2);imshow(angle(C2(C1(C0(x_est(:,:,KK(k)))))),[-pi/2,pi/2]);colorbar
    colormap(ax,'inferno')
    title('Retrieved sample phase','interpreter','latex','fontsize',12)
    drawnow;
end

%%
% =========================================================================
% Calculate errors
% =========================================================================
res = 0; ref = 0;
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



function [lam_1, lam_2, lam_3] = reg_param(iter,n_iters,alph,lams_start,lams_end)
% =========================================================================
% Setting the regularization parameter lam during the iterations using an
% sigmoid function.
% -------------------------------------------------------------------------
% Input:    - iter              : The current iteration number.
%           - n_iters           : Total iteration numbers.
%           - alph              : Sigmoid function parameter.
%           - lams_start        : Initial regularization parameter.
%           - lams_end          : Final regularization parameter.
% Output:   - lam_1,lam_2,lam_3 : The current regularization parameter for the three dimensions.
% =========================================================================
lam_1 = (lams_start(1)-lams_end(1))*1./(1+exp(alph*(iter/n_iters - 1/2))) + lams_end(1);
lam_2 = (lams_start(2)-lams_end(2))*1./(1+exp(alph*(iter/n_iters - 1/2))) + lams_end(2);
lam_3 = (lams_start(3)-lams_end(3))*1./(1+exp(alph*(iter/n_iters - 1/2))) + lams_end(3);
end


function H = transfunc_propagate(n1, n2, dist, pxsize, wavlen)
% =========================================================================
% Calculate the transfer function of the free-space diffraction.
% -------------------------------------------------------------------------
% Input:    - n1, n2   : The image dimensions (pixel).
%           - dist     : Propagation distance.
%           - pxsize   : Pixel (sampling) size.
%           - wavlen   : Wavelength of the light.
% Output:   - H        : Transfer function.
% =========================================================================

% sampling in the spatial frequency domain
k1 = pi/pxsize*(-1:2/n1:1-2/n1);
k2 = pi/pxsize*(-1:2/n2:1-2/n2);
[K2,K1] = meshgrid(k2,k1);

k = 2*pi/wavlen;    % wave number

ind = (K1.^2 + K2.^2 >= k^2);  % remove evanescent orders
K1(ind) = 0; K2(ind) = 0;

H = exp(1i*dist*sqrt(k^2-K1.^2-K2.^2));

end


function H = transfunc_imshift(n1,n2,s1,s2)
% =========================================================================
% Calculate the transfer function of the image shifting operation.
% -------------------------------------------------------------------------
% Input:    - n1, n2   : The image dimensions (pixel).
%           - s1, s2   : Shifts in the two dimension (pixel).
% Output:   - H        : Transfer function.
% =========================================================================
f1 = -n1/2:1:n1/2-1;
f2 = -n2/2:1:n2/2-1;
[u2,u1] = meshgrid(f2,f1);

H = exp(-1i*2*pi*(s1*u1/n1 + s2*u2/n2));

end


function w = Df(x)
% =========================================================================
% Calculate the 3D gradient (finite difference) of an input 3D datacube.
% -------------------------------------------------------------------------
% Input:    - x  : The input 3D datacube.
% Output:   - w  : The gradient (4D array).
% =========================================================================
if size(x,3) > 1
    w = cat(4, x(1:end,:,:) - x([2:end,end],:,:), ...
               x(:,1:end,:) - x(:,[2:end,end],:), ...
               x(:,:,1:end) - x(:,:,[2:end,end]));
else
    w = cat(4, x(1:end,:,:) - x([2:end,end],:,:), ...
               x(:,1:end,:) - x(:,[2:end,end],:), ...
               zeros(size(x(:,:,1))));
end
end


function u = DTf(w)
% =========================================================================
% Calculate the transpose of the gradient operator.
% -------------------------------------------------------------------------
% Input:    - w  : 4D array.
% Output:   - x  : 3D array.
% =========================================================================
u1 = w(:,:,:,1) - w([end,1:end-1],:,:,1);
u1(1,:,:) = w(1,:,:,1);
u1(end,:,:) = -w(end-1,:,:,1);

u2 = w(:,:,:,2) - w(:,[end,1:end-1],:,2);
u2(:,1,:) = w(:,1,:,2);
u2(:,end,:) = -w(:,end-1,:,2);

if size(w,3) > 1
    u3 = w(:,:,:,3) - w(:,:,[end,1:end-1],3);
    u3(:,:,1) = w(:,:,1,3);
    u3(:,:,end) = -w(:,:,end-1,3);
else
    u3 = 0;
end

u = u1 + u2 + u3;
end