function norm = normTVa(x,tau_s,tau_t)

grad = L(x);
norm = tau_s*sum(sqrt(sum(grad(:,:,:,:,1:2).^2,1)),[2,3,4,5]) ...
       + tau_t*sum(sqrt(sum(grad(:,:,:,:,3).^2,1)),[2,3,4,5]);

end

