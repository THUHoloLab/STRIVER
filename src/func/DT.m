function u = DT(w)

[n1,n2,n3,~] = size(w);

shift = circshift(w(:,:,:,1),[1,0,0]);
u1 = w(:,:,:,1) - shift;
u1(1,:,:) = w(1,:,:,1);
u1(n1,:,:) = -shift(n1,:,:);

shift = circshift(w(:,:,:,2),[0,1,0]);
u2 = w(:,:,:,2) - shift;
u2(:,1,:) = w(:,1,:,2);
u2(:,n2,:) = -shift(:,n2,:);

shift = circshift(w(:,:,:,3),[0,0,1]);
u3 = w(:,:,:,3) - shift;
u3(:,:,1) = w(:,:,1,3);
u3(:,:,n3) = -shift(:,:,n3);

u = u1 + u2 + u3;


end

