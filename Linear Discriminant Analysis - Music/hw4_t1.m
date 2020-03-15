clear; close all; clc;

files = dir('music/Test 1/');
minsize = 193138; % Initial runs to find this number
full_data = zeros(300, 32769*8);
counter = 0;
split_data = randperm(100);
split_data2 = randperm(100);
split_data3 = randperm(100);
for file = files(3:end)'
    [y, Fs] = audioread(strcat('music/Test 1/', file.name));
    disp(file.name)
    y = mean(y, 2);
    for ii=1:20
        yy = y(Fs*(ii*5):Fs*(ii*5 + 5));
        full_data(counter*20 + ii,:) = reshape(abs(spectrogram(yy)), [1 32769*8])';
    end
    counter = counter + 1;
end
full_data = full_data';
X = full_data(:, [split_data (100 + split_data2) (200 + split_data3)]);
X2 = [X(:, 51:100) X(:, 151:200) X(:, 251:300)];
X = [X(:, 1:50) X(:, 101:150) X(:, 201:250)];
%% Test 1 Band Classification
% Naruto, Polo G, Avicii
% playblocking(audioplayer(yy,Fs))
an = 50; nn = 50; rn = 50;
[U,S,V] = svd(X,'econ');
%%
songs = S*V';
feature=30;
avicii = songs(1:feature,1:an);
naruto = songs(1:feature,an+1:an+nn);
roddyrich = songs(1:feature,an+nn+1:an+nn+rn);

mu = mean(songs(1:feature,:));
ma = mean(avicii,2);
mn = mean(naruto,2);
mr = mean(roddyrich,2);

%% 
Sw=0;  % within class variances
for i=1:an
    Sw = Sw + (avicii(:,i)-ma)*(avicii(:,i)-ma)';
end
for i=1:nn
    Sw = Sw + (naruto(:,i)-mn)*(naruto(:,i)-mn)';
end
for i = 1:rn
    Sw = Sw + (roddyrich(:,i)-mr)*(roddyrich(:,i)-mr)';
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
vavicii = w'*avicii; vnaruto = w'*naruto; vroddyrich = w'*roddyrich;
vavicii1 = w1'*avicii; vnaruto1 = w1'*naruto; vroddyrich1 = w1'*roddyrich;
w_basis = [w w1];
figure(1)
plot(vavicii, vavicii1, 'ro'); hold on;
plot(vnaruto, vnaruto1, 'bo');
plot(vroddyrich, vroddyrich1, 'go'); hold off;
title('Training')
legend("Avicii", 'Naruto', 'Roddy Rich','Location', 'best')
xlabel("Eigenvector 1")
ylabel("Eigenvector 2")
%%
results = [vavicii vnaruto vroddyrich];
res2 = [vavicii1 vnaruto1 vroddyrich1];
tabl = [results' res2'];
trueRes = [repelem("Avicii", 50) repelem("Naruto", 50) repelem("Roddy Rich", 50)]';
%%
Mdl = fitcknn(tabl,trueRes,'NumNeighbors',45,'Standardize',1);
label = predict(Mdl, tabl);
labelz = repelem("",150);
for hh = 1:150
   labelz(hh) = label{hh};
end
c = sum(trueRes==labelz')/150;

%% TEST SET
an2 = 50; nn2 = 50; rn2 = 50; 
[U2,S2,V11] = svd(X2,'econ');
songs2 = S2*V11';
%%
feature=35;
avicii2 = songs2(1:feature,1:an2);
naruto2 = songs2(1:feature,an2+1:an2+nn2);
roddyrich2 = songs2(1:feature,an2+nn2+1:an2+nn2+rn2);

mu2 = mean(songs2(1:feature,:));
ma2 = mean(avicii2,2);
mn2 = mean(naruto2,2);
mr2 = mean(roddyrich2,2);

%% 
Sw2=0;  % within class variances
for i=1:an2
    Sw2 = Sw2 + (avicii2(:,i)-ma2)*(avicii2(:,i)-ma2)';
end
for i=1:nn2
    Sw2 = Sw2 + (naruto2(:,i)-mn2)*(naruto2(:,i)-mn2)';
end
for i = 1:rn2
    Sw2 = Sw2 + (roddyrich2(:,i)-mr2)*(roddyrich2(:,i)-mr2)';
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
vavicii2 = w2'*avicii2; vnaruto2 = w2'*naruto2; vroddyrich2 = w2'*roddyrich2;
vavicii12 = w12'*avicii2; vnaruto12 = w12'*naruto2; vroddyrich12 = w12'*roddyrich2;
results2 = [vavicii2 vnaruto2 vroddyrich2];
res22 = [vavicii12 vnaruto12 vroddyrich12];
tabl2 = [results2' res22'];
trueRes2 = [repelem("Avicii", 50) repelem("Naruto", 50) repelem("Roddy Rich", 50)]';
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
plot(vnaruto2, vnaruto12, 'bo');
plot(vroddyrich2, vroddyrich12, 'go'); hold off;
title('Testing')
legend("Avicii", 'Naruto', 'Roddy Rich','Location', 'best')
xlabel("Eigenvector 1")
ylabel("Eigenvector 2")