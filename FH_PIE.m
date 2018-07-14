addpath(genpath(pwd));
%% Parameter setting
conf.m=400;                              % Size of traing set
conf.upscale=4;                          % Magnification
conf.size=[160,120];                     % Size of HR image
pca_conf.DR=50;                          % PCA dim of HR residual
pca_conf.DL=50;                          % PCA dim of LR residual
cca_conf.D=40;                           % Final dim by CCA
cca_conf.rcov1=1e-4;                     % Regularization para of view1
cca_conf.rcov2=1e-4;                     % Regularization para of view2
% Parameter setting for step1
step1.nCluster=16;                       % Number of clusters in step1
step1.K=300;                             % Neighborhood size in step1
step1.s=8;                               % Neighborhood size in step1
step1.o=6;                               % Edge length of image block
step1.ksvd_conf.iternum = 20;            % Iterations of ksvd training
step1.ksvd_conf.memusage = 'normal';     % 
step1.ksvd_conf.dictsize = 512;          % Dictionarys size
step1.ksvd_conf.Tdata = 3;               % maximal sparsity: TBD
% Parameter setting for step2
step2.nCluster=14;
step2.K=300;
step2.s=30;
step2.o=25;
step2.ksvd_conf.iternum = 20;
step2.ksvd_conf.memusage = 'normal';
step2.ksvd_conf.dictsize = 512;
step2.ksvd_conf.Tdata = 3;
%% Load data and divide the image
load('Data\PIE\Multi_PIE_4x_tiny.mat');
IpRa = data2patchesSet(IH-IL, step1.s, step1.o);
IpLa = data2patchesSet(IL, step1.s, step1.o);

%% Cluster 1
IpR=cell(step1.nCluster,1);
IpL=cell(step1.nCluster,1);
[cidx,cen]=kmeans(IpLa',step1.nCluster,'MaxIter',300);
cen=cen';
for i =1:step1.nCluster
    IpR{i}=(IpRa(:,cidx==i));
    IpL{i}=(IpLa(:,cidx==i));
end
clear IpRa IpLa
%% PCA
BR=cell(step1.nCluster,1); BL=cell(step1.nCluster,1);
XR=cell(step1.nCluster,1); XL=cell(step1.nCluster,1);
MPR=cell(step1.nCluster,1); MPL=cell(step1.nCluster,1);
for i = 1:step1.nCluster
    [br_T,xr_T,~,~,explained,mpr_T]=pca(IpR{i}');
    pidx=pca_conf.DR;
    br=br_T'; xr=xr_T'; mpr=mpr_T';
    xr=xr(1:pidx,:);
    br=br(1:pidx,:);
    BR{i}=br; XR{i}=xr; MPR{i}=mpr;
    
    [bl_T,xl_T,~,~,explained,mpl_T]=pca(IpL{i}');
    pidx=pca_conf.DL;
    bl=bl_T'; xl=xl_T'; mpl=mpl_T';
    xl=xl(1:pidx,:);
    bl=bl(1:pidx,:);
    BL{i}=bl; XL{i}=xl; MPL{i}=mpl;
    
end

%% CCA
VR=cell(step1.nCluster,1); VL=cell(step1.nCluster,1);
MR=cell(step1.nCluster,1); ML=cell(step1.nCluster,1);
CR=cell(step1.nCluster,1); CL=cell(step1.nCluster,1);
for i = 1:step1.nCluster
    [vr, vl, mr, ml,~] = CCA(XR{i}, XL{i}, cca_conf.D, cca_conf.rcov1, cca_conf.rcov2);
    VR{i} = vr; MR{i} = mr;
    VL{i} = vl; ML{i} = ml;
    CR{i} = vr' * bsxfun(@minus, XR{i}, mr);
    CL{i} = vl' * bsxfun(@minus, XL{i}, ml);
end
%% Train the dic
DicR=cell(step1.nCluster,1); DicL=cell(step1.nCluster,1);
for i = 1:step1.nCluster
    step1.ksvd_conf.samples = size(CL{i},2);
    step1.ksvd_conf.data = CL{i};
    [dic_l, gamma] = ksvd(step1.ksvd_conf);
    DicL{i}=dic_l;
    DicR{i} = (CR{i} * gamma')/(full(gamma * gamma'));
end
clear CR CL XR XL
%% to find res
ILS=zeros(size(IL,1),size(IL,2),conf.m);
for i = 1 : conf.m
    i
    aIL=IL(:,:,i);
    [aIpL,prow,pcol] = overlapCut2vector(aIL, step1.s, step1.o);
    aIpR=zeros(size(aIpL,1),size(aIpL,2));
    for j =1:prow*pcol
        ipl=aIpL(:,j);
        distance = sum((cen-repmat(ipl,1,size(cen,2))).^2); % 2-norm
        [~,class]=min(distance);
        xpl = BL{class}*bsxfun(@minus,ipl,MPL{class});
        cpl=VL{class}'*(xpl-ML{class});
        [cpr,~,~] = neighbor2(cpl,DicL{class},DicR{class},step1.K);
        xpr=pinv(VR{class}')*cpr+MR{class};
        ipr=BR{class}'*xpr+MPR{class};
        aIpR(:,j)=ipr;
    end
    
    img1 = patches2image_vector(aIpR,prow,pcol, step1.s, step1.o);
    img2=img1+aIL;
    ILS(:,:,i) = img2;
end

ILSD = imresize(ILS,1/conf.upscale);
ILSD = imresize(ILSD, [conf.size(1),conf.size(2)]);

RH = IH - ILS;
RL = IL - ILSD;

RpHa = data2patchesSet(RH, step2.s, step2.o);
RpLa = data2patchesSet(RL, step2.s, step2.o);

clear IH IR IL IpR IpL

%% Cluster 2
RpH=cell(step2.nCluster,1);
RpL=cell(step2.nCluster,1);
[cidx2,cen2]=kmeans(RpLa',step2.nCluster,'MaxIter',300);
cen2=cen2';
for i =1:step2.nCluster
    RpH{i}=(RpHa(:,cidx2==i));
    RpL{i}=(RpLa(:,cidx2==i));
end




%% SR (Test)
IS=zeros(conf.size(1),conf.size(2),size(OtL,3));
PSNR=zeros(size(OtL,3),1);
SSIM=zeros(size(OtL,3),1);
for i = 1:size(OtL,3)
    i
    TL=OtL(:,:,i);
    TH=OtH(:,:,i);
    [TpL,prow,pcol] = overlapCut2vector(TL, step1.s, step1.o);
    TpR=zeros(size(TpL,1),size(TpL,2));
    for j =1:prow*pcol
        tpl=TpL(:,j);
        distance = sum((cen-repmat(tpl,1,size(cen,2))).^2); % 2-norm
        [~,class]=min(distance);
        xtl = BL{class}*bsxfun(@minus,tpl,MPL{class});
        ctl=VL{class}'*(xtl-ML{class});
        [ctr,~,~] = neighbor2(ctl,DicL{class},DicR{class},step1.K);
        xpr=pinv(VR{class}')*ctr+MR{class};
        tpr=BR{class}'*xpr+MPR{class};
        TpR(:,j)=tpr;
    end
    img = patches2image_vector(TpR,prow,pcol, step1.s, step1.o);
    ims=img+TL;
    
    imsd = imresize(ims,1/conf.upscale);
    imsd = imresize(imsd, [conf.size(1),conf.size(2)]);
    
    rl=TL-imsd;
    [rpl,prow,pcol] = overlapCut2vector(rl, step2.s, step2.o);
    rph=zeros(size(rpl,1),size(rpl,2));
    for j = 1:prow*pcol
        arpl=rpl(:,j);
        distance2 = sum((cen2-repmat(arpl,1,size(cen2,2))).^2); % 2-norm
        [~,class]=min(distance2);
        [arph,~,~] = neighbor2(arpl,RpL{class},RpH{class},step2.K);
        rph(:,j)=arph;
    end
    Rh = patches2image_vector(rph,prow,pcol, step2.s, step2.o);
    ims2 = ims+Rh;
    IS(:,:,i)=ims2;
    PSNR(i)=psnr(TH,ims2);
    SSIM(i)=ssim(TH,ims2);
end