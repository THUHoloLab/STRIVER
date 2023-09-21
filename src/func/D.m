function w = D(x)

[n1,n2,n3] = size(x);
w = zeros(n1,n2,n3,3);

w(:,:,:,1) = x - circshift(x,[-1,0,0]);
w(n1,:,:,1) = 0;
w(:,:,:,2) = x - circshift(x,[0,-1,0]);
w(:,n2,:,2) = 0;
w(:,:,:,3) = x - circshift(x,[0,0,-1]);
w(:,:,n3,3) = 0;

end

