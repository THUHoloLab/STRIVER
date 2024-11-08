function img_out = padimage(img, bias, size_out)

img_out = zeros(size_out);
[n1,n2] = size(img);

img_out(bias(1)+1:bias(1)+n1,bias(2)+1:bias(2)+n2) = img;


end