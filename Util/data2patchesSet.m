function P = data2patchesSet(imageSet, s, overlap)
[r, c, n] = size(imageSet);
P=[];
for i= 1 : n
    [B,~,~] = overlapCut2vector(imageSet(:,:,i),s,overlap);
    P=[P, B];
end