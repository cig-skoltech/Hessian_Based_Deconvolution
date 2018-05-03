function [x,P,R,iter,L]=denoiseHessSDual(y,lambda,varargin)

%This method uses the Hessian Spectral norm regularizer in the form of a compound
%regularizer with a box constraint and can be applied only to 2D data.
%x: denoised image
%P,R : dual variables
%L: Lipschitz constant
%iter: number of iterations for getting to the solution.
%bc: Boundary conditions for the differential operators
%    ('reflexive'|'circular'|'zero')

%Author: stamatis.lefkimmiatis@epfl.ch (Biomedical Imaging Group)

[maxiter,L,tol,optim,verbose,img,bounds,P,R,bc]=process_options(varargin,...
  'maxiter',100,'L',128/1.25,'tol',1e-4,'optim','fgp','verbose',false,...
  'img',[],'bounds',[-inf +inf],'P',zeros([size(y) 2]),'R',...
  zeros(size(y)),'bc','reflexive');

if isempty(L)
  L=Lipschitz(y)/1.25;% Lipschitz constant for the Compound Operator
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
    Q=R;
    for i=1:maxiter
      K=y-lambda/2*(AdjLaplacianOp2D(Q,bc)+AdjTV2Op2D(F,bc));
      %Pnew=F+step*lambda/2*TV2Op2D(y-lambda/2*(AdjLaplacianOp2D(Q)+
      %                                                    AdjTV2Op2D(F)));
      %step=1/(L1*(lambda/2)^2)==>
      %Pnew=F+(1/(L1*lambda/2))*TV2Op2D(y-lambda/2*(AdjLaplacianOp2D(Q)+
      %                                                    AdjTV2Op2D(F)));
      Pnew=F+(1/(L*lambda/2))*TV2Op2D(project(K,bounds),bc);
      Pnew=projectL2(Pnew);
      
      %Rnew=Q+step*lambda/2*LaplacianOp2D(y-lambda/2*(AdjLaplacianOp2D(Q)+
      %                                                    AdjTV2Op2D(F)));
      %step=1/(L2*(lambda/2)^2)==>
      %Rnew=Q+(1/(L2*lambda/2))*LaplacianOp2D(y-lambda/2*
      %                               (AdjLaplacianOp2D(Q)+AdjTV2Op2D(F)));
      Rnew=Q+(1/(L*lambda/2))*LaplacianOp2D(project(K,bounds),bc);
      Rnew=projectL1(Rnew);
      
      %relative error
      re=(norm(Pnew(:)-P(:))+norm(Rnew(:)-R(:)))/(norm(Pnew(:))+norm(Rnew(:)));
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      tnew=(1+sqrt(1+4*t^2))/2;
      F=Pnew+(t-1)/tnew*(Pnew-P);
      P=Pnew;
      Q=Rnew+(t-1)/tnew*(Rnew-R);
      R=Rnew;
      t=tnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda/2*(AdjLaplacianOp2D(R,bc)+AdjTV2Op2D(P,bc));
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          ISNR=20*log10(norm(y-img,'fro')/norm(x-img,'fro'));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
        else
          k=y-lambda/2*(AdjLaplacianOp2D(R,bc)+AdjTV2Op2D(P,bc));
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
      
      K=y-lambda/2*(AdjLaplacianOp2D(R,bc)+AdjTV2Op2D(P,bc));
      %Pnew=P+step*lambda/2*TV2Op2D(y-lambda/2*(AdjLaplacianOp2D(R)+
      %                                                    AdjTV2Op2D(P)));
      %step=1/(L1*(lambda/2)^2)==>
      %Pnew=P+(1/(L1*lambda/2))*TV2Op2D(y-lambda/2*(AdjLaplacianOp2D(R)+
      %                                                    AdjTV2Op2D(P)));
      Pnew=P+(1/(L*lambda/2))*TV2Op2D(project(K,bounds),bc);
      Pnew=projectL2(Pnew);
      
      %Rnew=R+step*lambda/2*LaplacianOp2D(y-lambda/2*(AdjLaplacianOp2D(Q)+
      %                                                    AdjTV2Op2D(F)));
      %step=1/(L2*(lambda/2)^2)==>
      %Rnew=R+(1/(L2*lambda/2))*LaplacianOp2D(y-lambda/2*
      %                               (AdjLaplacianOp2D(Q)+AdjTV2Op2D(F)));
      Rnew=R+(1/(L*lambda/2))*LaplacianOp2D(project(K,bounds),bc);
      Rnew=projectL1(Rnew);
      
      %relative error
      re=(norm(Pnew(:)-P(:))+norm(Rnew(:)-R(:)))/(norm(Pnew(:))+norm(Rnew(:)));
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      P=Pnew;
      R=Rnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda/2*(AdjLaplacianOp2D(R,bc)+AdjTV2Op2D(P,bc));
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          ISNR=20*log10(norm(y-img,'fro')/norm(x-img,'fro'));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
        else
          k=y-lambda/2*(AdjLaplacianOp2D(R,bc)+AdjTV2Op2D(P,bc));
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

x=project(y-lambda/2*(AdjLaplacianOp2D(R,bc)+AdjTV2Op2D(P,bc)),bounds);


function Lf=LaplacianOp2D(f,bc)

%Compute Lf (Apply to image f the Laplacian Operator)
%Lf will also be an image (2D image)
Lf=(2*f-2*shift(f,[-1,0],bc)-2*shift(f,[0,-1],bc)+shift(f,[-2,0],bc)+...
  shift(f,[0,-2],bc));

function LaR=AdjLaplacianOp2D(R,bc)

LaR=(2*R-2*shiftAdj(R,[-1,0],bc)-2*shiftAdj(R,[0,-1],bc)+...
  shiftAdj(R,[-2,0],bc)+shiftAdj(R,[0,-2],bc));

function Uf=TV2Op2D(f,bc)

[r,c]=size(f);

fd=(2*shift(f,[0,-1],bc)-2*shift(f,[-1,0],bc)+shift(f,[-2,0],bc)+...
  -shift(f,[0,-2],bc));
fxy=(f-shift(f,[0,-1],bc)-shift(f,[-1,0],bc)+shift(f,[-1,-1],bc));

%Compute Uf (Apply to image f the TV-like Operator U)
%Uf will be a 3D image
Uf=zeros(r,c,2);
Uf(:,:,1)=fd;%fd=fxx-fyy;
Uf(:,:,2)=2*fxy;

function UaP=AdjTV2Op2D(P,bc)

P1=P(:,:,1);
P1=(2*shiftAdj(P1,[0,-1],bc)-2*shiftAdj(P1,[-1,0],bc)+...
  shiftAdj(P1,[-2,0],bc)-shiftAdj(P1,[0,-2],bc));
P2=2*P(:,:,2);
P2=(P2-shiftAdj(P2,[0,-1],bc)-shiftAdj(P2,[-1,0],bc)+...
  shiftAdj(P2,[-1,-1],bc));

UaP=P1+P2;

function PB=projectL1(B)

%Check which matrices don't belong to the linf-norm ball.
%For those matrices normalize their values by PB=B/max(1,abs(B))

PB=B./max(1,abs(B));

function PB=projectL2(B)

%Check which cubes (matrices NxMx2) don't belong to the l2-norm ball.
%For those cubes normalize their values by
%PB=B(:,:,1)/max(1,sqrt(B(:,:,1)^2+B(:,:,2)^2),
%B(:,:,2)/max(1,sqrt(B(:,:,1)^2+B(:,:,2)^2)

%K=max(1,sqrt(sum(B.^2,3)));
%PB=B./repmat(K,[1 1 2]);
PB=B./repmat(max(1,sqrt(sum(B.^2,3))),[1 1 2]);

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

Lf=fxx+fyy;%Laplacian of the image f
Of=sqrt((fxx-fyy).^2+4*fxy.^2);% Amplitude of the Orientation vector

Hnorm=sum(sum(0.5*(abs(Lf)+Of)));%Sum of the Hessian spectral radius
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
Op_eig=abs(fft2(hxx+hyy)).^2+abs(fft2(hxx-hyy)).^2+4*abs(fft2(hxy)).^2;
L=max(Op_eig(:));
