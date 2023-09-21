function prox = proxTVa(x,gamma,tau_s,tau_t,iter)

lambda_s = tau_s*gamma;
lambda_t = tau_t*gamma;

lambda_s = tau_s;
lambda_t = tau_t;

t_prev = 1;

[~,n1,n2,n3] = size(x);
grad_next = zeros(2,n1,n2,n3,3);
grad_prev = zeros(2,n1,n2,n3,3);
temp = zeros(2,n1,n2,n3,3);

for i = 1:iter
    grad_next = temp + 1/8/gamma*L(x - gamma*LT(temp));
    deno = zeros(2,n1,n2,n3,3);
    if tau_s > 0 && tau_t > 0
        deno(1,:,:,:,1) = 1/lambda_s*max(lambda_s,sqrt(grad_next(1,:,:,:,1).^2 + grad_next(2,:,:,:,1).^2));
        deno(1,:,:,:,2) = 1/lambda_s*max(lambda_s,sqrt(grad_next(1,:,:,:,2).^2 + grad_next(2,:,:,:,2).^2));
        deno(1,:,:,:,3) = 1/lambda_t*max(lambda_t,sqrt(grad_next(1,:,:,:,3).^2 + grad_next(2,:,:,:,3).^2));
        deno(2,:,:,:,1) = deno(1,:,:,:,1);
        deno(2,:,:,:,2) = deno(1,:,:,:,2);
        deno(2,:,:,:,3) = deno(1,:,:,:,3);
        grad_next = grad_next./deno;
    elseif tau_s > 0 && tau_t == 0
        deno(1,:,:,:,1) = 1/lambda_s*max(lambda_s,sqrt(grad_next(1,:,:,:,1).^2 + grad_next(2,:,:,:,1).^2));
        deno(1,:,:,:,2) = 1/lambda_s*max(lambda_s,sqrt(grad_next(1,:,:,:,2).^2 + grad_next(2,:,:,:,2).^2));
        deno(2,:,:,:,1) = deno(1,:,:,:,1);
        deno(2,:,:,:,2) = deno(1,:,:,:,2);
        grad_next(:,:,:,:,1:2) = grad_next(:,:,:,:,1:2)./deno(:,:,:,:,1:2);
        grad_next(:,:,:,:,3) = zeros(2,n1,n2,n3);
    elseif tau_t > 0 && tau_s == 0
        deno(1,:,:,:,3) = 1/lambda_t*max(lambda_t,sqrt(grad_next(1,:,:,:,3).^2 + grad_next(2,:,:,:,3).^2));
        deno(2,:,:,:,3) = deno(1,:,:,:,3);
        grad_next(:,:,:,:,3) = grad_next(:,:,:,:,3)./deno(:,:,:,:,3);
        grad_next(:,:,:,:,1:2) = zeros(2,n1,n2,n3,2);
    elseif tau_s == 0 && tau_t == 0
        grad_next = zeros(2,n1,n2,n3,3);
    end
    t_next = (1+sqrt(1+4*t_prev^2))/2;
    temp = grad_next + (t_prev-1)/t_next*(grad_next-grad_prev);
    grad_prev = grad_next;
    t_prev = t_next;
end  

prox = x - gamma*LT(grad_next);

end

