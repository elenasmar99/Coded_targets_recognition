clear
clc

%Make a new table that contains every target (no cells)
load('labels9.mat');
fileNames =labels9{:,1}; 
for i=1:size(labels9,1)
    row = cell2mat(labels9{i,2:end}');
    targetBBs{i,1} = row;
end

labels9 = table(fileNames,targetBBs,'VariableNames',{'imageFilenames','targets'});

save('labels_oneTarget9.mat','labels9')

%Load positive samples.

load('labels_oneTarget9.mat');

%Select the bounding boxes for stop signs from the table.

positiveInstances = labels9(:,1:2);

%Add the image folder to the MATLAB path.

imDir = fullfile('C:\','Users\','Elena\','Desktop\','tcd\','+p');
addpath(imDir);

%Specify the folder for negative images.

negativeFolder = fullfile('C:\','Users\','Elena\','Desktop\','tcd\','-n');

%Create an imageDatastore object containing negative images.

negativeImages = imageDatastore(negativeFolder);

%%% augmentedImageSource
%Train a cascade object detector called 'stopSignDetector.xml' using HOG features. NOTE: The command can take several minutes to run.

trainCascadeObjectDetector('TargetDetector.xml',positiveInstances, ...
    negativeFolder,'FalseAlarmRate',0.1,'NumCascadeStages',5);
%Use the newly trained classifier to detect a target in an image.

detector = ψιρψλε('TargetDetector.xml');

%Read the test image.

img = imread('test.jpg');
img=img(:,:,3);
imshow(img)

%Detect a target sign.

bbox = step(detector,img);

img_labeled = insertObjectAnnotation(img,'rectangle',bbox,'target');
imshow(img_labeled)

%Crop the produced image

rows=size(bbox,1);

for i=1:rows 
    
    bb=bbox(i,:);
    x=bb(1);
    y=bb(2);
    w=bb(3);
    h=bb(4);
    
    Ic=img(y:y+h,x:x+w,:);
    
    %Find the circles in the cropped image    
    r=round((min(w,h))/2);
    figure
    imshow(Ic)
    
    [centers,radii,metric] = imfindcircles(Ic,[floor(r/2),r],'ObjectPolarity','dark','Sensitivity',0.99);
        
    %Select best
    
    [~,max_index] = max(metric);
    
    hold on
    viscircles(centers(max_index,:),radii(max_index));
    hold off

     
end


%Read the test image.
T22=imread('target22.jpg');
T22=T22(:,:,3);

%Detect a target sign.
bb2 = step(detector,T22);
img_labeled2 = insertObjectAnnotation(T22,'rectangle',bb2,'target');
figure
imshow(img_labeled2)

%Crop the produced image
    x2=bb2(1);
    y2=bb2(2);
    w2=bb2(3);
    h2=bb2(4);
    
    Ic2=T22(y2:y2+h2,x2:x2+w2,:);
    figure
    imshow(Ic2)
    
%Find the circles in the cropped image    
    r2=round((min(w2,h2))/2);
    
    [ce,ra,me] = imfindcircles(Ic2,[floor(r2/2),r2],'ObjectPolarity','dark','Sensitivity',0.99);
        
    %Select best
    
    [~,max_index2] = max(me);
    
    hold on
    viscircles(ce(max_index2,:),ra(max_index2));
    hold off

%Crop bbox
Ic2=Ic2((ce(2)-ra):(ce(2)+ra),(ce(1)-ra):(ce(1)+ra),:);
figure
imshow(Ic2)


B=bbox;
x3=B(1);
y3=B(2);
w3=B(3);
h3=B(4);
Ic3=img(y3:y3+h3,x3:x3+w3,:);
imshow(Ic3)

%Find the circles in the cropped image    
    r3=round((min(w3,h3))/2);
    
    [cee,raa,mee] = imfindcircles(Ic3,[floor(r3/2),r3],'ObjectPolarity','dark','Sensitivity',0.99);
        
    %Select best
    
    [~,max_index3] = max(mee);
    
    hold on
    viscircles(cee(max_index3,:),raa(max_index3));
    hold off

    
%Crop image
Ic3=Ic3((cee(2)-raa):(cee(2)+raa),(cee(1)-raa):(cee(1)+raa),:);
figure
imshow(Ic3)


%Correlation between the two images
max_corr=-1;
for theta=(15:-0.5:-15)
    disp(theta)
    
    %B(i) = imrotate(Ic2,theta,'loose',bbox);
    B=imrotate(Ic2,theta);
    s=size(B);
    N=imresize(Ic3,s);
    
    corr=corr2(B,N);
    
    if corr>max_corr
        max_theta=theta;
    end
    max_corr=max(max_corr,corr)
    
end

disp(max_corr)
disp(max_theta)
