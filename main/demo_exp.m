% ========================================================================
% Introduction
% ========================================================================
% This code provides a simple demonstration of dynamic phase retrieval via
% spatiotemporal total variation regularization.
%
% Reference:
%   - Y. Gao and L. Cao, “Motion-resolved, reference-free holographic 
%     imaging via spatiotemporally regularized inversion," Optica 11, 
%     32-41 (2024).
%
% Author: Yunhui Gao (gyh21@mails.tsinghua.edu.cn)
% =========================================================================
%%
% =========================================================================
% Data generation
% =========================================================================
clear;clc;
close all;

% load functions
addpath(genpath('utils'))
addpath(genpath('../src'))

% select experiment data
exp_num = 13; grp_num = 97; frame_start = 20;

% load calibration data
foldername = ['../data/exp/E',num2str(exp_num),'/G',num2str(grp_num)];
load([foldername,'/calib/calib_diffuser.mat'],'diffuser','bias_1','bias_2','sizeout_1','sizeout_2','prefix','params');
load([foldername,'/calib/calib_shift.mat'],'shifts')

% experiment settings
K = 10;     % number of measurements
shifts_ref = shifts(:,frame_start);
shifts = shifts(:,frame_start:frame_start+K-1); % calibrated lateral displacement
if sum(isnan(shifts(:)))
    error('Shifts include nan values.');
end
shifts = shifts - shifts_ref;
img_obj = padimage(im2double(imread([foldername,'/',prefix,num2str(frame_start),'.bmp'])),...
        [bias_1,bias_2],[sizeout_1,sizeout_2]);

% select the area of interest
fprintf('Please select the area of interest ... ')
figure
[temp,rect_aoi_image] = imcrop(img_obj);
if rem(size(temp,1),2) == 1
    rect_aoi_image(4) = rect_aoi_image(4) - 1;
end
if rem(size(temp,2),2) == 1
    rect_aoi_image(3) = rect_aoi_image(3) - 1;
end
close
fprintf('Selected.\n')

nullpixels_1 = 50;
nullpixels_2 = 50;

% spatial dimension of the image
M1 = round(rect_aoi_image(4));
M2 = round(rect_aoi_image(3));
y = nan(M1,M2,K);

% spatial dimension of the diffuser
MM1 = M1 + 2*nullpixels_2;
MM2 = M2 + 2*nullpixels_2;
diffusers = nan(MM1,MM2,K);
rect_aoi_diffuser = [rect_aoi_image(1)-nullpixels_2, rect_aoi_image(2)-nullpixels_2,...
    rect_aoi_image(3)+2*nullpixels_2, rect_aoi_image(4)+2*nullpixels_2];

% load the measurement data
mask  = ones(size(im2double(imread([foldername,'/',prefix,num2str(frame_start),'.bmp']))));
masks = zeros(M1,M2,K);
fprintf('Loading raw data ... ')
for k = 1:K
    if k>1; fprintf(repmat('\b',1,9)); end
    fprintf('%03d / %03d',k,K);
    img_obj = padimage(im2double(imread([foldername,'/',prefix,num2str(frame_start+(k-1)),'.bmp'])),...
        [bias_1,bias_2],[sizeout_1,sizeout_2]);
    img_obj  = abs(imshift(img_obj, shifts(1,k), shifts(2,k)));
    y(:,:,k) = imcrop(img_obj,rect_aoi_image);
    mask_tmp = padimage(mask,[bias_1,bias_2],[sizeout_1,sizeout_2]);
    mask_tmp = abs(imshift(mask_tmp, shifts(1,k), shifts(2,k)));
    masks(:,:,k) = imcrop(mask_tmp,rect_aoi_image);
    diff = imshift(diffuser, shifts(1,k), shifts(2,k));
    diffusers(:,:,k) = imcrop(abs(diff),rect_aoi_diffuser) .* exp(1i*imcrop(angle(diff),rect_aoi_diffuser));
end
fprintf('\n')

% display the data
figure
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])
for k = 1:K
    subplot(1,3,1),imshow(y(:,:,k),[]);title('Measurement')
    subplot(1,3,2),imshow(abs(diffusers(:,:,k)),[]);title('Diffuser amplitude')
    subplot(1,3,3),imshow(masks(:,:,k),[0,1]);title('FOV mask')
    drawnow;
end

%%
% =========================================================================
% Forward model
% =========================================================================
% spatial dimension of the sample
N1 = M1 + 2*nullpixels_2 + 2*nullpixels_1;  
N2 = M2 + 2*nullpixels_2 + 2*nullpixels_1;

% pre-calculate the transfer functions for diffraction modeling
HQ1 = fftshift(transfunc_propagate(N1,N2, params.dist_1,params.pxsize,params.wavlen)); % forward propagation
HQ2 = fftshift(transfunc_propagate(M1+2*nullpixels_2,M2+2*nullpixels_2, params.dist_2,params.pxsize,params.wavlen)); % forward propagation

% forward model
Q1  = @(x)   ifft2(fft2(x).*HQ1);                   % free-space propagation operator from sample to diffuser
Q1H = @(x)   ifft2(fft2(x).*conj(HQ1));             % Hermitian operator of Q1
C1  = @(x)   imgcrop(x,nullpixels_1);               % image cropping operator
C1T = @(x)   zeropad(x,nullpixels_1);               % transpose operator of C1
M   = @(x,k) x.*diffusers(:,:,k);                   % diffuser modulation operator
MH  = @(x,k) x.*conj(diffusers(:,:,k));             % Hermitian operator of M
Q2  = @(x)   ifft2(fft2(x).*HQ2);                   % free-space propagation operator from diffuser to sensor
Q2H = @(x)   ifft2(fft2(x).*conj(HQ2));             % Hermitian operator of Q2
C2  = @(x)   imgcrop(x,nullpixels_2);               % image cropping operator
C2T = @(x)   zeropad(x,nullpixels_2);               % transpose operator of C2
S   = @(x,k) x.*masks(:,:,k);                       % masking operator to avoid invalid pixels
ST  = @(x,k) x.*conj(masks(:,:,k));                 % transpose operator of S
A   = @(x,k) S(C2(Q2(M(C1(Q1(x)),k))),k);           % entire measurement operator
AH  = @(x,k) Q1H(C1T(MH(Q2H(C2T(ST(x,k))),k)));     % Hermitian operator of A

% =========================================================================
% Phase retrieval algorithm
% =========================================================================

K_recon = K;        % number of reconstructed frames

% algorithm settings
x_init = ones(M1+2*nullpixels_1+2*nullpixels_2,M2+2*nullpixels_1+2*nullpixels_2,K_recon);     % initial guess
lams_s = [1e-0, 2e-3];                          % regularization parameter (spatial)
lams_t = [5e-0, 2e-3];                          % regularization parameter (temporal)

alph = 10;              % hyperparameter for tuning regularization weights
gam  = 2;               % step size (see the paper for details)
n_iters    = 200;       % number of iterations (main loop)
n_subiters = 3;         % number of subiterations (proximal update)

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