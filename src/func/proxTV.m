function x = proxTV(v,gamma,n_iters)

global lam_1; global lam_2; global lam_3;


[n1,n2,n3] = size(v);
w = zeros(n1,n2,n3,3);
w_prev = zeros(n1,n2,n3,3);
z = zeros(n1,n2,n3,3);

for t = 1:n_iters
    w = z + 1/12/gamma*D(v - gamma*DT(z));
    w(:,:,:,1) = min(abs(w(:,:,:,1)),lam_1).*exp(1i*angle(w(:,:,:,1)));
    w(:,:,:,2) = min(abs(w(:,:,:,2)),lam_2).*exp(1i*angle(w(:,:,:,2)));
    w(:,:,:,3) = min(abs(w(:,:,:,3)),lam_3).*exp(1i*angle(w(:,:,:,3)));
    
    z = w + t/(t+3)*(w-w_prev);
    w_prev = w;
end  

x = v - gamma*DT(w);

end

