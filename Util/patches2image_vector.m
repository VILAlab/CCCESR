function img = patches2image_vector(P,prow,pcol,s,overlap)
I = zeros(prow*(s-overlap)+overlap,pcol*(s-overlap)+overlap);
flag = zeros(prow*(s-overlap)+overlap,pcol*(s-overlap)+overlap);
for i=1:prow
    for j=1:pcol
        for t=1:s
            I((s-overlap)*(i-1)+1:(s-overlap)*(i-1)+s,(s-overlap)*(j-1)+t) = ...
                I((s-overlap)*(i-1)+1:(s-overlap)*(i-1)+s,(s-overlap)*(j-1)+t) + ...
                P(s*(t-1)+1:s*t,(i-1)*pcol+j);
            flag((s-overlap)*(i-1)+1:(s-overlap)*(i-1)+s,(s-overlap)*(j-1)+t)=...
                flag((s-overlap)*(i-1)+1:(s-overlap)*(i-1)+s,(s-overlap)*(j-1)+t) + 1;
        end
    end
end
img(:,:) = I./flag;
end