clear; close all; clc;

files = dir('music/Test 2/');
minsize = 193138; % Initial runs to find this number
full_data = zeros(300, 32769*8);
counter = 0;
split_data = randperm(100);
for file = files(3:end)'
    [y, Fs] = audioread(strcat('music/Test 2/', file.name));
    disp(file.name)
    y = mean(y, 2);
    for ii=1:20
        yy = y(Fs*(ii*5):Fs*(ii*5 + 5));
        full_data(counter*20 + ii,:) = reshape(abs(spectrogram(yy)), [1 32769*8])';
    end
    counter = counter + 1;
end
full_data = full_data';
X = full_data(:, [split_data 100 + split_data 200 + split_data]);
X2 = [X(:, 51:100) X(:, 151:200) X(:, 251:300)];
X = [X(:, 1:50) X(:, 101:150) X(:, 201:250)];
%% Test 1 Band Classification
% Naruto, Polo G, Avicii
% playblocking(audioplayer(yy,Fs))
an = 50; nn = 50; rn = 50;
feature=38; 
[U,S,V] = svd(X,'econ');
songs = S*V';
avicii = songs(1:feature,1:an);
chainsmokers = songs(1:feature,an+1:an+nn);
tiesto = songs(1:feature,an+nn+1:an+nn+rn);

mu = mean(songs(1:feature,:));
ma = mean(avicii,2);
mn = mean(chainsmokers,2);
mr = mean(tiesto,2);

%% 
Sw=0;  % within class variances
for i=1:an
    Sw = Sw + (avicii(:,i)-ma)*(avicii(:,i)-ma)';
end
for i=1:nn
    Sw = Sw + (chainsmokers(:,i)-mn)*(chainsmokers(:,i)-mn)';
end
for i = 1:rn
    Sw = Sw + (tiesto(:,i)-mr)*(tiesto(:,i)-mr)';
end
Sb = (ma-mu)*(ma-mu)' + (mn-mu)*(mn-mu)' + (mr-mu)*(mr-mu)';
[V2,D] = eig(Sb,Sw);
[~,ind] = max(abs(diag(D)));
w=V2(:,ind); w=w/norm(w,2);

%%
D33 = D;
D33(:,ind) = [];
D33(ind, :) = [];
[~, indx] = max(abs(diag(D33)));
w1 = V2(:, indx + 1); w1 = w1/norm(w1,2);
%%
vavicii = w'*avicii; vchainsmokers = w'*chainsmokers; vtiesto = w'*tiesto;
vavicii1 = w1'*avicii; vchainsmokers1 = w1'*chainsmokers; vtiesto1 = w1'*tiesto;
figure(1)
plot(vavicii, vavicii1, 'ro'); hold on;
plot(vchainsmokers, vchainsmokers1, 'bo');
plot(vtiesto, vtiesto1, 'go'); hold off;
title('Training')
legend("Avicii", 'Chainsmokers', 'Tiesto','Location', 'best')
xlabel("Eigenvector 1")
ylabel("Eigenvector 2")
%%
results = [vavicii vchainsmokers vtiesto];
res2 = [vavicii1 vchainsmokers1 vtiesto1];
tabl = [results' res2'];
trueRes = [repelem("Avicii", 50) repelem("Chainsmokers", 50) repelem("Tiesto", 50)]';
%%
Mdl = fitcknn(tabl,trueRes,'NumNeighbors',45, 'Standardize', 1);
label = predict(Mdl, tabl);
labelz = repelem("",150);
for hh = 1:150
   labelz(hh) = label{hh};
end
c = sum(trueRes==labelz')/150;

%% TEST SET
an2 = 50; nn2 = 50; rn2 = 50;
feature=38; 
[U2,S2,V11] = svd(X2,'econ');
songs2 = S2*V11';
avicii2 = songs2(1:feature,1:an2);
chainsmokers2 = songs2(1:feature,an2+1:an2+nn2);
tiesto2 = songs2(1:feature,an2+nn2+1:an2+nn2+rn2);

mu2 = mean(songs2(1:feature,:));
ma2 = mean(avicii2,2);
mn2 = mean(chainsmokers2,2);
mr2 = mean(tiesto2,2);

%% 
Sw2=0;  % within class variances
for i=1:an2
    Sw2 = Sw2 + (avicii2(:,i)-ma2)*(avicii2(:,i)-ma2)';
end
for i=1:nn2
    Sw2 = Sw2 + (chainsmokers2(:,i)-mn2)*(chainsmokers2(:,i)-mn2)';
end
for i = 1:rn2
    Sw2 = Sw2 + (tiesto2(:,i)-mr2)*(tiesto2(:,i)-mr2)';
end
Sb2 = (ma2-mu2)*(ma2-mu2)' + (mn2-mu2)*(mn2-mu2)' + (mr2-mu2)*(mr2-mu2)';
[V22,D2] = eig(Sb2,Sw2);
[lambda,ind2] = max(abs(diag(D2)));
w2=V22(:,ind2); w2=w2/norm(w2,2);
%%
D332 = D2;
D332(:,ind2) = [];
D332(ind2, :) = [];
[~, indx2] = max(abs(diag(D332)));
w12 = V22(:, indx2 + 1); w12 = w12/norm(w12,2);
%%
vavicii2 = w2'*avicii2; vchainsmokers2 = w2'*chainsmokers2; vtiesto2 = w2'*tiesto2;
vavicii12 = w12'*avicii2; vchainsmokers12 = w12'*chainsmokers2; vtiesto12 = w12'*tiesto2;
results2 = [vavicii2 vchainsmokers2 vtiesto2];
res22 = [vavicii12 vchainsmokers12 vtiesto12];
tabl2 = [results2' res22'];
trueRes2 = [repelem("Avicii", 50) repelem("Chainsmokers", 50) repelem("Tiesto", 50)]';
%%
label2 = predict(Mdl, tabl2);
labelz2 = repelem("",150);
for hh = 1:150
   labelz2(hh) = label2{hh};
end
c2 = sum(trueRes2==labelz2')/150;
%%
figure(2)
plot(vavicii2, vavicii12, 'ro'); hold on;
plot(vchainsmokers2, vchainsmokers12, 'bo');
plot(vtiesto2, vtiesto12, 'go'); hold off;
title('Testing')
legend("Avicii", 'Chainsmokers', 'Tiesto','Location', 'best')
xlabel("Eigenvector 1")
ylabel("Eigenvector 2")