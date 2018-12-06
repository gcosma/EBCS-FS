%Copyright (c) 2017, Sadegh Salesi and Georgina Cosma. All rights reserved.

function [fitness,acc,nfeat]=svm(X,Y,pop)

FeatIndex = find(pop==1);
trainset = X(:,[FeatIndex]);
trainlabel=Y;
SVMModel=fitcsvm(trainset,trainlabel,'KernelFunction','rbf');
CVSVMModel = crossval(SVMModel);
loss=kfoldLoss(CVSVMModel);

ntotal_feat=size(X,2);
nfeat=numel(FeatIndex);
acc=1-loss;
fitness=-acc;  %classic fitness function
%fitness=(0.2*(nfeat/ntotal_feat))-0.8*acc; %proposed fitness function