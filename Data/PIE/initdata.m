function [IH,IL] = initdata(rowH,colH,rowL,colL)

file_path =  '.\Fig\';% ͼ���ļ���·��  
% file_path =  '.\GroundTruth\';
img_path_list = dir(strcat(file_path,'*.png'));%��ȡ���ļ���������jpg��ʽ��ͼ��  
img_num = length(img_path_list);%��ȡͼ�������� 

IH=zeros(rowH,colH,500);
IL=zeros(rowH,colH,500);

for i=1:500
    image_name = img_path_list(i).name;% ͼ����  
    img =  imread(strcat(file_path,image_name));  
    img=im2double(img);
    if ndims(img)==3 
        img=rgb2gray(img);
    end
    imh=imresize(img,[rowH,colH],'bicubic');
    iml=imresize(imh,[rowL,colL],'bicubic');
    iml=imresize(iml,[rowH,colH],'bicubic');
    IH(:,:,i)=imh;
    IL(:,:,i)=iml;
end