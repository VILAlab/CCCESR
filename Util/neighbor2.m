function [CTH,W,neighborhood] = neighbor2(CT,CL,CH,K);  
% compute weight matrix U by reconstructing nearest appearance neighbors (nan) of XT in XS  
% CT :  N by T matrix of T features, each colomn representing a N-D feature vector  
% CL :  N by S matrix of S features, each colomn representing a N-D feature vector  
% CH :  M by S matrix of S features, each colomn representing a M-D feature vector (M>N)  
% K :  # of nearest appearance neighbors, deciding by Euclidean distance  
%  
% CT :  M by T matrix of T features, each colomn representing a M-D feature vector  
% W :  K by T weight matrix  
% neighborhood : K by T matrix indicating the neighbors of each feature vector  
[~,N] = size(CT);  
[M,~] = size(CH);  
  
neighborhood = zeros(K,N);
for i = 1:N  
    tempCT = CT(:,i);  
    distance = sum((CL-repmat(tempCT,1,size(CL,2))).^2); % 2-norm
    [~,index] = sort(distance,2);  
    neighborhood(:,i) = index(2:(K+1));  % 第i行是第i个测试样本的K个近邻分别在CL中的索引
end  
tol=1e-4; % regularlizer in case constrained fits are ill conditioned  
  

W = zeros(K,N);  
for j=1:N  
    z = CL(:,neighborhood(:,j))-repmat(CT(:,j),1,K); % shift ith pt to origin  
    C = z'*z;                                        % local covariance  
    if trace(C)==0  
        C = C + eye(K,K)*tol;                   % regularlization  
    else  
        C = C + eye(K,K)*tol*trace(C);  
    end  
     W(:,j) = C\ones(K,1);                           % solve C*w=1  
    W(:,j) = W(:,j)/sum(W(:,j));                  % enforce sum(w)=1  
end

CTH=zeros(M,N);
for j = 1:N  
    CTH(:,j) = CH(:,neighborhood(:,j))*W(:,j);  
end  