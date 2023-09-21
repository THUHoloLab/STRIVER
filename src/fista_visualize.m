function [x,J_vals,E_vals,runtimes] = fista_visualize(x_init,F,dF,R,proxR,gamma,n_iters,opts)
% initialization
x = x_init;
z = x;
J_vals = NaN(n_iters+1,1);
E_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
if isa(opts.lams,'function_handle')
        opts.lams(0);
end

J_vals(1) = F(x)+R(x);
if isa(opts.errfunc,'function_handle')
    E_vals(1) = opts.errfunc(z);
end


timer = tic;
figure
set(gcf,'unit','normalized','position',[0.2,0.2,0.6,0.5])
for iter = 1:n_iters

    if isa(opts.lams,'function_handle')
        opts.lams(iter);
    end
    
    % proximal gradient update
    x_next = proxR(z - gamma*dF(z),gamma);
    J_vals(iter+1) = F(x)+R(x);
    z = x_next + (iter/(iter+3))*(x_next - x);
    
    % record runtime
    runtimes(iter) = toc(timer);
    
    % calculate error metric
    if isa(opts.errfunc,'function_handle')
        E_vals(iter+1) = opts.errfunc(z);
    end
    
    % display status
    if opts.verbose
        fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
                iter, J_vals(iter+1), gamma, runtimes(iter));
    end
    
    x = x_next;
    
    subplot(1,2,1),imshow(abs(x(:,:,1)),[]);colorbar
    subplot(1,2,2),imshow(angle(x(:,:,1)),[]);colorbar
    drawnow;
end

end

