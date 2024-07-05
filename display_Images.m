clc 
clear
close all

imageFiles = dir('imagery\*.tif');
bandImages =[];
for i = 1: length(imageFiles)
    if contains(imageFiles(i).name , "183333") && ~contains(imageFiles(i).name, "A") &&  ~contains(imageFiles(i).name, "Fmask") 
        bandImages = [bandImages; imageFiles(i)];
    end
end

for i = 1:length(bandImages)
    fileName = fullfile(bandImages(i).folder, bandImages(i).name);
    img = imread(fileName);
    figure;
    imshow(img);
    title([bandImages(i).name]);
end
