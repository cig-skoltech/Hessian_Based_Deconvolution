function run_demo


f=double(imread('peppers256.png'));
f=f/255;

h=fspecial('gaussian',9,6); % psf kernel
b=imfilter(f,h,'conv','circular'); % In the forward model it is assumed 
%that the convolution is circular. 

sigma=sqrt(var(b(:))/10^2);%noise which corresponds to a BSNR of 20 dBs.

y=b+sigma*randn(size(f));%blurred+noisy observation

% Algorithmic Parameters (See deconvHessFDual.m and deconvHessSDual.m for a
% description).
options={'x_init',[],'iter',100,'den_iter',10,'verbose',true,'showfig',false,'optim','mfista','den_thr',1e-3,'deconv_thr',1e-5,'den_optim','fgp','bounds',[0 1],'img',f,'bc','reflexive'};

HF=deconvHessFDual(y,h,6e-4,options{:});
HS=deconvHessSDual(y,h,7e-4,options{:});

figure('name','Observation');
imshow(y,[]);

figure('name','Hessian-Frobenius Regularizer');
imshow(HF,[]);

figure('name','Hessian-Spectral Regularizer');
imshow(HS,[]);
