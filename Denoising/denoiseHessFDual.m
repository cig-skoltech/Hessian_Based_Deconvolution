function [x,P,iter,L]=denoiseHessFDual(y,lambda,varargin)

%This method uses the Frobenious norm Hessian-based regularizer with a box
%constraint and can be applied only to 2D data.
%x: denoised image
%P: dual variables
%L: Lipschitz constant
%iter: number of iterations for getting to the solution.
%bc: Boundary conditions for the differential operators
%    ('reflexive'|'circular'|'zero')

%Author: stamatis.lefkimmiatis@epfl.ch (Biomedical Imaging Group)

[maxiter,L,tol,optim,verbose,img,bounds,P,bc]=process_options(varargin,...
  'maxiter',100,'L',64/1.25,'tol',1e-4,'optim','fgp','verbose',false,...
  'img',[],'bounds',[-inf +inf],'P',zeros([size(y) 3]),'bc','reflexive');

if isempty(L)
  L=Lipschitz(y)/1.25;% Lipschitz constant for the Operator
end

count=0;
flag=false;

if verbose
  fprintf('******************************************\n');
  fprintf('**  Denoising with Hessian Regularizer  **\n');
  fprintf('******************************************\n');
  fprintf('#iter     relative-dif   \t fun_val         Duality Gap    \t   ISNR\n')
  fprintf('====================================================================\n');
end
switch optim
  case 'fgp'
    t=1;
    F=P;
    for i=1:maxiter
      K=y-lambda*AdjTV2Op2D(F,bc);
      %Pnew=F+step*lambda*TV2Op2D(y-lambda*AdjTV2Op2D(F));
      %step=1/(L*(lambda)^2)==>
      %Pnew=F+(1/(L*lambda))*TV2Op2D(y-lambda*AdjTV2Op2D(F));
      Pnew=F+(1/(L*lambda))*TV2Op2D(project(K,bounds),bc);
      Pnew=projectL2(Pnew);
      
      %relative error
      re=(norm(Pnew(:)-P(:)))/norm(Pnew(:));
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      tnew=(1+sqrt(1+4*t^2))/2;
      F=Pnew+(t-1)/tnew*(Pnew-P);
      P=Pnew;
      t=tnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda*AdjTV2Op2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          ISNR=20*log10(norm(y-img,'fro')/norm(x-img,'fro'));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
        else
          k=y-lambda*AdjTV2Op2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val,dual_gap);
        end
      end
      
      if count >=5
        flag=true;
        iter=i;
        break;
      end
    end
    
  case 'gp'
    
    for i=1:maxiter
      
      K=y-lambda*AdjTV2Op2D(P,bc);
      %Pnew=F+step*lambda/2*TV2Op2D(y-lambda*AdjTV2Op2D(F));
      %step=1/(L*(lambda)^2)==>
      %Pnew=F+(1/(L*lambda))*TV2Op2D(y-lambda*AdjTV2Op2D(F));
      Pnew=P+(1/(L*lambda))*TV2Op2D(project(K,bounds),bc);
      Pnew=projectL2(Pnew);
      
      %relative error
      re=(norm(Pnew(:)-P(:)))/norm(Pnew(:));
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      P=Pnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda*AdjTV2Op2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          ISNR=20*log10(norm(y-img,'fro')/norm(x-img,'fro'));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
        else
          k=y-lambda*AdjTV2Op2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val,dual_gap);
        end
      end
      
      if count >=5
        flag=true;
        iter=i;
        break;
      end
    end
end

if ~flag
  iter=maxiter;
end

x=project(y-lambda*AdjTV2Op2D(P,bc),bounds);



function Uf=TV2Op2D(f,bc)

[r,c]=size(f);

fxx=(f-2*shift(f,[-1,0],bc)+shift(f,[-2,0],bc));
fyy=(f-2*shift(f,[0,-1],bc)+shift(f,[0,-2],bc));
fxy=(f-shift(f,[0,-1],bc)-shift(f,[-1,0],bc)+shift(f,[-1,-1],bc));

%Compute Uf (Apply to image f the TV-like Operator U)
%Uf will be a 3D image
Uf=zeros(r,c,3);
Uf(:,:,1)=fxx;
Uf(:,:,2)=fyy;
Uf(:,:,3)=sqrt(2)*fxy;

function UaP=AdjTV2Op2D(P,bc)

P1=P(:,:,1);
P1=(P1-2*shiftAdj(P1,[-1,0],bc)+shiftAdj(P1,[-2,0],bc));
P2=P(:,:,2);
P2=(P2-2*shiftAdj(P2,[0,-1],bc)+shiftAdj(P2,[0,-2],bc));
P3=sqrt(2)*P(:,:,3);
P3=(P3-shiftAdj(P3,[0,-1],bc)-shiftAdj(P3,[-1,0],bc)+...
  shiftAdj(P3,[-1,-1],bc));

UaP=P1+P2+P3;


function PB=projectL2(B)

%Check which cubes (matrices NxMx3) don't belong to the l2-norm ball.
%For those cubes normalize their values by
%PB=B(:,:,1)/max(1,sqrt(B(:,:,1)^2+B(:,:,2)^2+B(:,:,3)^2),
%B(:,:,2)/max(1,sqrt(B(:,:,1)^2+B(:,:,2)^2+B(:,:,3)^2),
%B(:,:,3)/max(1,sqrt(B(:,:,1)^2+B(:,:,2)^2+B(:,:,3)^2),

%K=max(1,sqrt(sum(B.^2,3)));
%PB=B./repmat(K,[1 1 2]);
PB=B./repmat(max(1,sqrt(sum(B.^2,3))),[1 1 3]);

function Px=project(x,bounds)
lb=bounds(1);%lower box bound
ub=bounds(2);%upper box bound

if isequal(lb,-Inf) && isequal(ub,Inf)
  Px=x;
elseif isequal(lb,-Inf) && isfinite(ub)
  x(x>ub)=ub;
  Px=x;
elseif isequal(ub,Inf) && isfinite(lb)
  x(x<lb)=lb;
  Px=x;
else
  x(x<lb)=lb;
  x(x>ub)=ub;
  Px=x;
end

function [Q,Hnorm]=cost(y,f,lambda,bc)

fxx=(f-2*shift(f,[-1,0],bc)+shift(f,[-2,0],bc));
fyy=(f-2*shift(f,[0,-1],bc)+shift(f,[0,-2],bc));
fxy=(f-shift(f,[0,-1],bc)-shift(f,[-1,0],bc)+shift(f,[-1,-1],bc));

Hnorm=sqrt(fxx.^2+fyy.^2+2*fxy.^2);% Amplitude of the Orientation vector

Hnorm=sum(Hnorm(:));%Sum of the Hessian spectral radius
Q=0.5*norm(y-f,'fro')^2+lambda*Hnorm;

function Q=dualcost(y,f,bounds)
r=f-project(f,bounds);
Q=0.5*(sum(r(:).^2)+sum(y(:).^2)-sum(f(:).^2));


function L=Lipschitz(y)
%Finds the Lipschitz constant for the dual function.

[r,c]=size(y);

hxx=[1 -2 1 0 0]';% Dxx Operator
hxx=padarray(hxx,[0 2]);
hyy=hxx'; %Dyy Operator
hxy=[1 -1 0;-1 1 0;0 0 0];%Dxy Operator
hxy=padarray(hxy,[1 1]);
center=[3 3];

hxx(r,c)=0;hxy(r,c)=0;hyy(r,c)=0;

hxx=circshift(hxx,1-center);
hyy=circshift(hyy,1-center);
hxy=circshift(hxy,1-center);

%Operator eigenvalues
Op_eig=abs(fft2(hxx)).^2+abs(fft2(hyy)).^2+2*abs(fft2(hxy)).^2;
L=max(Op_eig(:));
