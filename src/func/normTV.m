function n = normTV(x)

global lam_1; global lam_2; global lam_3;

g = D(x);
n = lam_1 * norm1(g(:,:,:,1)) + lam_2 * norm1(g(:,:,:,2)) + lam_3 * norm1(g(:,:,:,3));

function v = norm1(x)
    v = norm(x(:),1);
end

end
