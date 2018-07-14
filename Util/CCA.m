function [A,B,m1,m2,D]=CCA(H1,H2,dim,rcov1,rcov2)
% H1 and H2 are dxN matrices containing samples colwise.
% dim is the desired dimensionality of CCA space.
% r is the regularization of autocovariance for computing the correlation.
% A and B are the transformation matrix for view 1 and view 2.
% m1 and m2 are the mean for view 1 and view 2.
% D is the vector of singular values.

if nargin<4
  rcov1=0; rcov2=0;
end

[d1,N] =size(H1);
[d2,~] =size(H2);
% Remove mean.
m1 = mean(H1,2); H1 = bsxfun(@minus,H1,m1);
m2 = mean(H2,2); H2 = bsxfun(@minus,H2,m2);

S11 = (H1*H1')/(N-1)+rcov1*eye(d1); S22 = (H2*H2')/(N-1)+rcov2*eye(d2); 
S12 = (H1*H2')/(N-1);
[V1,D1] = eig(S11); [V2,D2] = eig(S22);
% For numerical stability.
D1 = diag(D1); idx1 = find(D1>1e-12); D1 = D1(idx1); V1 = V1(:,idx1);
D2 = diag(D2); idx2 = find(D2>1e-12); D2 = D2(idx2); V2 = V2(:,idx2);

K11 = V1*diag(D1.^(-1/2))*V1';
K22 = V2*diag(D2.^(-1/2))*V2';
T = K11*S12*K22;
[U,D,V] = svd(T,0);
D = diag(D);

A = K11*U(:,1:dim);
B = K22*V(:,1:dim);
D = D(1:dim);


