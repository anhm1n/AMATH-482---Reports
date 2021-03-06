clear; clc; close all;

load('fashion_mnist.mat')
%%
X_train = im2double(X_train);
X_test = im2double(X_test);

X_train = reshape(X_train,[60000 28 28 1]);
X_train = permute(X_train,[2 3 4 1]);

X_test = reshape(X_test,[10000 28 28 1]);
X_test = permute(X_test,[2 3 4 1]);

X_valid = X_train(:,:,:,1:5000);
X_train = X_train(:,:,:,5001:end);

y_valid = categorical(y_train(1:5000))';
y_train = categorical(y_train(5001:end))';
y_test = categorical(y_test)';
%%
rng(123)
layers = [imageInputLayer([28 28 1])
        fullyConnectedLayer(700)
        reluLayer
        fullyConnectedLayer(500)
        reluLayer
        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',50,...
    'InitialLearnRate',1e-2, ...
    'ValidationData',{X_valid,y_valid}, ...
    'Verbose',true)
net = trainNetwork(X_train,y_train,layers,options);

%% Confusion for training
figure(1)
y_pred = classify(net,X_train);
plotconfusion(y_train,y_pred)

figure(3)
y_pred2 = classify(net,X_valid);
plotconfusion(y_valid,y_pred2)
%% Test classifier
figure(3)
y_pred3 = classify(net,X_test);
plotconfusion(y_test,y_pred3)