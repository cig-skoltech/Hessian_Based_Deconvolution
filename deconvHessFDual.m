function [x,x_init,fun_val,ISNR]=deconvHessFDual(y,h,lambda,varargin)

%Image deconvolution with a Hessian Frobenius Norm regularizer 
%using a majorization-minimization approach. Minimization of the majorizer 
%is performed using a proximal splitting method.
%
% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values Description
% =========================================================================
% y             Noisy blured image.
% h             Blur kernel (PSF) (Circular convolution is assumed)
% lambda        Regularization penalty.
% ======================== OPTIONAL INPUT PARAMETERS ======================
% Parameters    Values Description
% =========================================================================
% img           Original Image. (For the compution of the ISNR improvement)
% x_init        Initial guess for x. (Default: [])
% iter          Number of iterations (Default: 100)
% den_iter      Number of denoising iterations (Default: 15)
% den_thr       Stopping threshold for denoising (Default:1e-3)
% den_optim     Which gradient-based method use for the denoising problem.
%               {'fgp'|'gp'}. (Fast Gradient Projection or Gradient
%               Projection) (Default: 'fgp')
% deconv_thr    Stopping threshold for deconvolution (Default:1e-5)
% optim         Which version of fista to use for minimizing the majorizer
%               in ith iteration. {'fista'|'mfista'}  (Default: 'mfista')
% verbose       If verbose is set on then info for each iteration are
%               printed on screen. (Default: false)
% showfig       If showfig is set on the result of the deconvolution in
%               each iteration is shown on screen. (Default: false)
% bounds        Minimize the Objective with a box constraint on the
%               solution (Default: [-inf +inf])
% bc            Boundary conditions for the differential operators.
%               {'reflexive'|'circular'|'zero'} (Default: 'reflexive')
% =========================================================================

%Author: stamatis.lefkimmiatis@epfl.ch (Biomedical Imaging Group)


[x_init,iter,den_iter,verbose,showfig,optim,den_thr,deconv_thr,bc,...
    den_optim,bounds,img]=process_options(varargin,'x_init',[],'iter',100,...
    'den_iter',10,'verbose',false,'showfig',false,'optim','mfista',...
    'den_thr',1e-3,'deconv_thr',1e-5,'bc','reflexive','den_optim','fgp',...
    'bounds',[-inf +inf],'img',[]);

% Initializations
[hr,hc]=size(h);
[yr,yc]=size(y);

%zero padding h to match the image size.
hzp=h;
hzp(yr,yc)=0;

%Circular shift of the PSF so as to have consistent results by applying
%either one of the following 2 operations on an image x (hzp is the zero
%padded psf)
%1)imfilter(x,h,'conv','circular')
%2)ifft2(fft2(x).*fft2(hzp));

%============================ IMPORTANT NOTE ==============================
%Matlab's imfilter picks as the origin of h its central value, i.e
% chy=floor((hr+1)/2) and chx=floor((hc+1)/2). Therefore without circular
% shifting hzp, the above 2 operations are not consistent.
%==========================================================================

hzp=circshift(hzp,[-floor(hr/2),-floor(hc/2)]);

%x0 Initialization
H=fft2(hzp);
Y=fft2(y);

if isequal(x_init,'wiener')
  sigma=wmad(y,'db7');% estimation of the noise standard deviation.
  % Initialization with Wiener deconvolution filter
  x_init= real(ifft2((conj(H).*Y)./(abs(H).^2+10*sigma^2/var(y(:)))));
  ISNR=20*log10(norm(y-img,'fro')/norm(x_init-img,'fro'));
  fprintf('ISNR for Wiener regularization : %f\n',ISNR);
elseif isempty(x_init)
    x_init=y;
end


L=max(max(abs(H).^2))+eps;%  norm(H*H',2)
x=x_init;

Y=fft2(y);
%FISTA Minimization
f=x;
t=1;

fun_val=zeros(iter,1);
ISNR=zeros(iter,1);
P=zeros([size(y) 3]);
count=0;
Lipconst=[];
if verbose 
fprintf('\t\t***********************************\n');
fprintf('\t\t**  Deconvolution with FISTA     **\n');
fprintf('\t\t***********************************\n');
fprintf('#iter       fun-val    den-iter   relative-dif       ISNR\n')
fprintf('======================================================================\n');
end
for i=1:iter
  
  b=f+(1/L)*ifft2(conj(H).*Y-abs(H).^2.*fft2(f));
  [z,P,deniter,Lipconst]=denoiseHessFDual(b,lambda/L,'P',P,'tol',...
    den_thr,'maxiter',den_iter,'optim',den_optim,'L',Lipconst,...
    'bounds',bounds,'bc',bc);
  
  fun_val(i)=cost(y,H,z,lambda,bc);
  if isequal(optim,'mfista') && i > 1
    if (fun_val(i)>fun_val(i-1))
      xnew=x;
      fun_val(i)=fun_val(i-1);
    else
      xnew=z;
    end
  else
    xnew=z;
  end
  
  %Updating t and f
  tnew=(1+sqrt(1+4*t^2))/2;
  f=xnew+(t/tnew)*(z-xnew)+(t-1)/tnew*(xnew-x);
   
  re=norm(xnew-x,'fro')/norm(x,'fro');%relative error
  if (re<deconv_thr)
    count=count+1;
  else
    count=0;
  end
  
  if verbose
    if ~isempty(img)
        ISNR(i)=20*log10(norm(y-img,'fro')/norm(xnew-img,'fro'));
    % printing the information of the current iteration
        fprintf('%3d \t %10.5f \t %3d \t %10.5f \t %10.5f\n',i,fun_val(i),deniter,re,ISNR(i));
    else
        fprintf('%3d \t %10.5f \t %3d \t %10.5f\n',i,fun_val(i),deniter,re);
    end
  end
  
  x=xnew;
  t=tnew;
  if showfig
    fh=figure(1);
    figure(fh);
    msg=['iteration: ' num2str(i) ' ,ISNR: ' num2str(ISNR(i))];
    set(fh,'name',msg);imshow(x,[]);
  end
  
  if count >=5
    break;
  end
  
end


function [Q,Hnorm]=cost(y,H,f,lambda,bc)

fxx=(f-2*shift(f,[-1,0],bc)+shift(f,[-2,0],bc));
fyy=(f-2*shift(f,[0,-1],bc)+shift(f,[0,-2],bc));
fxy=(f-shift(f,[0,-1],bc)-shift(f,[-1,0],bc)+shift(f,[-1,-1],bc));


Hnorm=sqrt(fxx.^2+fyy.^2+2*fxy.^2);

Hf=real(ifft2(H.*fft2(f)));
Hnorm=sum(Hnorm(:));%Sum of the Hessian Frobenious norm

Q=0.5*norm(y-Hf,'fro')^2+lambda*Hnorm;


