function [P,prow,pcol] = overlapCut2vector(img,s,overlap)
[m,n] = size(img);
prow = (m-overlap)/(s-overlap);  
pcol = (n-overlap)/(s-overlap);  
P = zeros(s*s,prow*pcol);
for i = 1:prow
    for j = 1:pcol
        index = (i-1)*pcol + j;
        p = img((i-1)*(s-overlap)+1:(i-1)*(s-overlap)+s,(j-1)*(s-overlap)+1:(j-1)*(s-overlap)+s);
        P(:,index) = p(:);
    end
end
end