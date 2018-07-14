function [IH,IL] = initdata(rowH,colH,rowL,colL)

file_path =  '.\Fig\';% 图像文件夹路径  
% file_path =  '.\GroundTruth\';
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有jpg格式的图像  
img_num = length(img_path_list);%获取图像总数量 

IH=zeros(rowH,colH,500);
IL=zeros(rowH,colH,500);

for i=1:500
    image_name = img_path_list(i).name;% 图像名  
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