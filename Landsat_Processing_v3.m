%Universal Landsat Processing Script
%Tom Bell
%Oct 11, 2020
%WHAT A TIME TO BE ALIVE...
%Only works for Landsat images in the NW hemisphere for now...

%Set Path/Row Number
PR = '042036';
% Set Directories
%Directory where folders of Landsat images resides
path = '/Volumes/Landsat_C2/';
%path = '/Users/tbell/Desktop/';
DEMpath = '/Users/tbell/Desktop/';
directory = [path PR '/ZSFiles/'];
cd(directory)
folds = dir(directory);
folds(ismember({folds.name}, {'.', '..', '.DS_Store'})) = [];
files = {folds.name}';
bad_files = NaN(1,1);
RB = 1;

for j = 1:size(files,1)
    
    clc
    disp(['Running ' char(files(j))])
    %Finds individual band files
    ins = [directory char(files(j)) '/'];
    cd(ins)
    folds2 = dir(ins);
    folds2(ismember({folds2.name}, {'.', '..', '.DS_Store'})) = [];
    files2 = {folds2.name}';
    
    %Verify all files present
    fileList = dir('*.tif');
    Name = fileList(1).name;
    
    Prefix = Name(1:17);
    PrefixDate = [Prefix files{j}];
    
    DateTIFs = dir([PrefixDate '*.tif']);
    TotTIFs = dir('*.tif');
    
    if size(DateTIFs,1) < 10 || size(TotTIFs,1) > 10
        disp([files{j} ' Incorrect # of Bands in Folder']);
        pause(2)
        bad_files(RB,1) = str2double(files{j});
        RB = RB + 1;
        continue
    end
    
    Sensor = str2double(Name(4));
    
    %Determine sensor
    if Sensor == 4 || Sensor == 5
        SName = 1;
    elseif Sensor == 7
        SName = 2;
    elseif Sensor == 8
        SName = 3;
    end
    
    %Opens one band file
    fileB1 = dir('*band1.tif');
    [A,R1] = geotiffread(fileB1(1).name);
    info = geotiffinfo(fileB1(1).name);
    
    %Set up UTM coordinate rasters
    height = info.Height;
    width = info.Width;
    [rows,cols] = meshgrid(1:height,1:width);
    [Xutm,Yutm] = pix2map(info.RefMatrix, rows, cols);
    
    %Create coordinate file in lat/lon
    [Yll,Xll] = utm2ll(Xutm,Yutm,info.Zone);
    coords = [Yll(:) Xll(:)];
    
    %Find existing DEMs for Landsat image
    coordsR = round(coords);
    coordsRU = abs(unique(coordsR,'rows'));
    coordsRU1 = coordsRU;
    coordsRU1(:,2) = coordsRU1(:,2) + 1;
    coordsRU2 = coordsRU;
    coordsRU2(:,1) = coordsRU1(:,1) - 1;
    coordsRUt = [coordsRU;coordsRU1;coordsRU2];
    coordsRUt = unique(coordsRUt,'rows');
    
    fileEx = ones(size(coordsRUt,1),1);
    
    for ii = 1:size(coordsRUt,1)
        try
            Aj = geotiffinterp([DEMpath 'DEM/ASTGTMV003_N' num2str(coordsRUt(ii,1)) 'W' num2str(coordsRUt(ii,2)) '_dem.tif'],coords(:,2),coords(:,1));
        catch
            fileEx(ii,1) = 0;
        end
    end
    
    fileEx = logical(fileEx);
    coordsRUt = coordsRUt(fileEx,:);
    
    %Generate landmask from DEMs
    DEM = zeros(size(Xutm));
    clc
    disp('Generating Landmask')
    
    for i = 1:size(coordsRUt,1)
        Ai = geotiffinterp([DEMpath 'DEM/ASTGTMV003_N' num2str(coordsRUt(i,1)) 'W' num2str(coordsRUt(i,2)) '_dem.tif'],coords(:,2),coords(:,1));
        Ai(Ai > 0) = 1;
        Aii = reshape(Ai,size(Xutm)); %makes sure aii has same dimensions as Xutm
        DEM = cat(3,DEM,Aii); %stacks new landmask layer on top of digital elevation model
        DEM = nansum(DEM,3); %sums along 3rd dimension? Not sure why you couldnt just use Aii  
    end
    
    clear Ai Aii
    
    DEM(DEM < 0) = 0;
    DEM = DEM';
    DEM = logical(DEM);
    
    %Dilate DEM by 1 pixel
    SE = strel('square',3);
    DEMd = imdilate(DEM,SE);
    
    clear A ans cols coords coordsR coordsRU DateTIFs fileEx fileList height ...
        rows SE width TotTIFs
    
    %Classify based on sensor
    if SName < 3
        
        %Read in Landsat band files
        fileB1 = dir('*band1.tif');
        A1 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band2.tif');
        A2 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band3.tif');
        A3 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band4.tif');
        A4 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band5.tif');
        A5 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band7.tif');
        A6 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*pixel_qa.tif');
        A7 = geotiffread(fileB1(1).name);
        
        %Sets all negative values to NaN
        A1(A1 < 0) = NaN;
        A2(A2 < 0) = NaN;
        A3(A3 < 0) = NaN;
        A4(A4 < 0) = NaN;
        A5(A5 < 0) = NaN;
        A6(A6 < 0) = NaN;
        A7(A7 < 0) = NaN;
        
        %Applies water mask to all files
        A1(DEMd) = 0;
        A2(DEMd) = 0;
        A3(DEMd) = 0;
        A4(DEMd) = 0;
        A5(DEMd) = 0;
        A6(DEMd) = 0;
        A7(DEMd) = 0;
        
        %Merges masked bands into one data matrix
        tmp = cat(3,A1,A2,A3,A4,A5,A6);
        tmp = double(tmp);
        
        %Normalizes matrix and convers to unsigned 8 bit
        C = sum(tmp,3);
        C1 = tmp./C;
        C1 = uint8(round(C1*255));
        
        %Finds possible cloud pixels and converts to unsigned 8 bit
        Cl = find(A7 ~= 0 & A7 ~= 66 & A7 ~= 68 & A7 ~= 130 & A7 ~= 132 & A7 ~= 72 & A7 ~= 136);
        CSLC = A7 == 1;
        A7(Cl) = 255;
        A7(A7 < 255) = 0;
        A7(A7 == 255) = 1;
        Clouds = logical(A7);
        
        clear A5 A6 A7
        
        %Restructures bands for classification
        D = double(C1);
        
        D1 = D(:,:,1);
        D2 = D(:,:,2);
        D3 = D(:,:,3);
        D4 = D(:,:,4);
        D5 = D(:,:,5);
        D6 = D(:,:,6);
        
        D1 = D1(:);
        D2 = D2(:);
        D3 = D3(:);
        D4 = D4(:);
        D5 = D5(:);
        D6 = D6(:);
        
        DD = [D1 D2 D3 D4 D5 D6];
        
        clear D1 D2 D3 D4 D5 D6
        
        %Loads classifier file and predicts the class of each pixel
        clc
        disp('Running Classifier')
        load Dec_tree_kelp.mat
        Ynew = predict(tc,DD);
        
        %Organizes predictions into data matrix
        Ynewn = NaN(length(Ynew),1);
        
        for i = 1:length(Ynew)
            if Ynew{i} == 'a'
                Ynewn(i,1) = 1;
            elseif Ynew{i} == 'b'
                Ynewn(i,1) = 2;
            elseif Ynew{i} == 'c'
                Ynewn(i,1) = 1;
            else
                Ynewn(i,1) = 4;
            end
        end
        
        Ynewn = reshape(Ynewn,size(D,1),size(D,2));
        Ynewn(Clouds) = 3;
        Ynewn(CSLC) = 5;        
        
        
        %MESMA
        %Set Up Endmembers
        Kelp = [459; 556; 437; 1227];
        
        %Dilate DEM by many pixels for water endmembers
        SE = strel('square',200);
        DEMdw = imdilate(DEM,SE);
        DEMdw = DEMdw';
        
        Ynewnw = Ynewn';
        Ynewnw(DEMdw) = 2;
        Ynewnw = Ynewnw(:);
        
        W = find(Ynewnw == 1);
        
        if size(W,1) < 1000
            continue
        end
        
        Rand = randi(size(W,1),30,1);
        Ws = W(Rand);
        
        A1 = A1';
        A2 = A2';
        A3 = A3';
        A4 = A4';
        
        Image(:,1) = A1(:);
        Image(:,2) = A2(:);
        Image(:,3) = A3(:);
        Image(:,4) = A4(:);
        
        Image = double(Image);
        
        Water = Image(Ws,:);
        Water = Water';
        
        frac1 = NaN(size(Xutm,1),size(Xutm,2),30);
        frac2 = NaN(size(Xutm,1),size(Xutm,2),30);
        rmse = NaN(size(Xutm,1),size(Xutm,2),30);
        R = 1;
        
        clc
        disp('Running MESMA')
        for k = 1:30
            B = [Water(:,k) Kelp];
            [U,S,V] = svd(B,0);
            IS = V/S;
            em_inv = IS*transpose(U);
            F = em_inv*Image';
            model = F'*B';
            resids = (Image - model)/10000;
            rmse(:,:,R) = reshape(sqrt(mean(resids.^2,2)),size(Xutm,1),size(Xutm,2));
            frac1(:,:,R) = reshape(F(1,:),size(Xutm,1),size(Xutm,2));
            frac2(:,:,R) = reshape(F(2,:),size(Xutm,1),size(Xutm,2));
            R = R + 1;
            clc
            fprintf('Percent MESMA %d', round(100/30 * k));
        end
        
        clc
        disp('Exporting File')
        [minVals, PageIdx] = nanmin(rmse,[],3);
        [rows, cols] = ndgrid(1:size(rmse,1), 1:size(rmse,2));
        Zindex = sub2ind(size(rmse), rows, cols, PageIdx);
        Mes2 = frac2(Zindex);
        Mes2 = Mes2';
        
        KelpClass = Ynewn ~= 4;
        Mes2(KelpClass) = 0;
        Mes2(Mes2 < 0) = 0;
        Mes2 = Mes2 .* 100;
        Mes2 = round(Mes2);
        Mes2(Clouds) = -2;
        Mes2(CSLC) = -1;
        
        Mes2_16 = int16(Mes2);
        
        clear F W Water Rand R minVals model A1 A2 A3 A4 B C C1 Clouds cols CSLC ...
            DEM DEMd D DD DEMdw em_inv F frac1 frac2 rmse Image IS KelpClass ...
            Mes2 PageIdx rows Sensor SName I V Ws ZIndex Ynew Ynewn SLC_Class
        
        %Saves classfied file as geotiff
        outs = [path PR '/Mesma/' char(files{j}) '.tif'];
        UTMRef = 32600 + info.Zone;
        geotiffwrite(outs,Mes2_16,R1,'CoordRefSysCode',UTMRef)
        
    elseif SName == 3
        
        %Read in Landsat band files
        fileB1 = dir('*band2.tif');
        A1 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band3.tif');
        A2 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band4.tif');
        A3 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band5.tif');
        A4 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band6.tif');
        A5 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*band7.tif');
        A6 = geotiffread(fileB1(1).name);
        
        fileB1 = dir('*pixel_qa.tif');
        A7 = geotiffread(fileB1(1).name);
        
        %Sets all negative values to NaN
        A1(A1 < 0) = NaN;
        A2(A2 < 0) = NaN;
        A3(A3 < 0) = NaN;
        A4(A4 < 0) = NaN;
        A5(A5 < 0) = NaN;
        A6(A6 < 0) = NaN;
        A7(A7 < 0) = NaN;
        
        %Applies water mask to all files
        A1(DEMd) = 0;
        A2(DEMd) = 0;
        A3(DEMd) = 0;
        A4(DEMd) = 0;
        A5(DEMd) = 0;
        A6(DEMd) = 0;
        A7(DEMd) = 0;
        
        %Merges masked bands into one data matrix
        tmp = cat(3,A1,A2,A3,A4,A5,A6);
        tmp = double(tmp);
        
        %Normalizes matrix and convers to unsigned 8 bit
        C = sum(tmp,3);
        C1 = tmp./C;
        C1 = uint8(round(C1*255));
        
        %Finds possible cloud pixels and converts to unsigned 8 bit
        Cl = find(A7 ~= 0 & A7 ~= 1 & A7 ~= 322 & A7 ~= 386 & A7 ~= 834 & A7 ~= 898 & A7 ~= 1346 & A7 ~= 324 & A7 ~= 388 & A7 ~= 836 & A7 ~= 900 & A7 ~= 1348 & A7 ~= 352);
        A7(Cl) = 1352;
        A7(A7 < 1352) = 0;
        A7(A7 == 1352) = 1;
        A7 = uint8(A7);
        Clouds = logical(A7);
        
        clear A5 A6 A7
        
        %Restructures bands for classification
        D = double(C1);
        
        D1 = D(:,:,1);
        D2 = D(:,:,2);
        D3 = D(:,:,3);
        D4 = D(:,:,4);
        D5 = D(:,:,5);
        D6 = D(:,:,6);
        
        D1 = D1(:);
        D2 = D2(:);
        D3 = D3(:);
        D4 = D4(:);
        D5 = D5(:);
        D6 = D6(:);
        
        DD = [D1 D2 D3 D4 D5 D6];
        
        clear D1 D2 D3 D4 D5 D6
        
        %Loads classifier file and predicts the class of each pixel
        clc
        disp('Running Classifier')
        load Dec_tree_kelp8.mat
        Ynew = predict(tc8,DD);
        
        %Organizes predictions into data matrix
        Ynewn = NaN(length(Ynew),1);
        
        for i = 1:length(Ynew)
            if Ynew{i} == 'a'
                Ynewn(i,1) = 1;
            elseif Ynew{i} == 'b'
                Ynewn(i,1) = 2;
            elseif Ynew{i} == 'c'
                Ynewn(i,1) = 1;
            else
                Ynewn(i,1) = 4;
            end
        end
        
        Ynewn = reshape(Ynewn,size(D,1),size(D,2));
        Ynewn(Clouds) = 3;
        
        %MESMA
        %Set Up Endmembers
        Kelp = [459; 556; 437; 1227];
        
        %Dilate DEM by many pixels for water endmembers
        SE = strel('square',200);
        DEMdw = imdilate(DEM,SE);
        DEMdw = DEMdw';
        
        Ynewnw = Ynewn';
        Ynewnw(DEMdw) = 2;
        Ynewnw = Ynewnw(:);
        
        W = find(Ynewnw == 1);
        
        if size(W,1) < 1000
            continue
        end
        
        Rand = randi(size(W,1),30,1);
        Ws = W(Rand);
        
        A1 = A1';
        A2 = A2';
        A3 = A3';
        A4 = A4';
        
        Image(:,1) = A1(:);
        Image(:,2) = A2(:);
        Image(:,3) = A3(:);
        Image(:,4) = A4(:);
        
        Image = double(Image);
        
        Water = Image(Ws,:);
        Water = Water';
        
        frac1 = NaN(size(Xutm,1),size(Xutm,2),30);
        frac2 = NaN(size(Xutm,1),size(Xutm,2),30);
        rmse = NaN(size(Xutm,1),size(Xutm,2),30);
        
        R = 1;
        
        clc
        disp('Running MESMA')
        for k = 1:30
            B = [Water(:,k) Kelp];
            [U,S,V] = svd(B,0);
            IS = V/S;
            em_inv = IS*transpose(U);
            F = em_inv*Image';
            model = F'*B';
            resids = (Image - model)/10000;
            rmse(:,:,R) = reshape(sqrt(mean(resids.^2,2)),size(Xutm,1),size(Xutm,2));
            frac1(:,:,R) = reshape(F(1,:),size(Xutm,1),size(Xutm,2));
            frac2(:,:,R) = reshape(F(2,:),size(Xutm,1),size(Xutm,2));
            R = R + 1;
            clc
            fprintf('Percent MESMA %d', round(100/30 * k));
        end
        
        clc
        disp('Exporting File')
        [minVals, PageIdx] = nanmin(rmse,[],3);
        [rows, cols] = ndgrid(1:size(rmse,1), 1:size(rmse,2));
        Zindex = sub2ind(size(rmse), rows, cols, PageIdx);
        Mes2 = frac2(Zindex);
        Mes2 = Mes2';
        
        %Landsat 8 mesma correction
        Mes2 = -0.229 * Mes2.^2 + 1.449 * Mes2 - 0.018;
        
        KelpClass = Ynewn ~= 4;
        Mes2(KelpClass) = 0;
        Mes2(Mes2 < 0) = 0;
        Mes2 = Mes2 .* 100;
        Mes2 = round(Mes2);
        Mes2(Clouds) = -2;
        
        Mes2_16 = int16(Mes2);
        
        clear F W Water Rand R minVals model A1 A2 A3 A4 B C C1 Clouds cols CSLC ...
            DEM DEMd D DD DEMdw em_inv F frac1 frac2 rmse Image IS KelpClass ...
            Mes2 PageIdx rows Sensor SName I V Ws ZIndex Ynew Ynewn
        
        %Saves classfied file as geotiff
        outs = [path PR '/Mesma/' char(files{j}) '.tif'];
        UTMRef = 32600 + info.Zone;
        geotiffwrite(outs,Mes2_16,R1,'CoordRefSysCode',UTMRef)
    end
end