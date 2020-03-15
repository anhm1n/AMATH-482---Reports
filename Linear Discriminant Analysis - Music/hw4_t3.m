clear; close all; clc;

files = dir('music/Test 3/');
minsize = 193138; % Initial runs to find this number
full_data = zeros(32769*8, 300);
counter = 0;
split_data = randperm(100);
for file = files(3:end)'
    [y, Fs] = audioread(strcat('music/Test 3/', file.name));
    disp(file.name)
    y = mean(y, 2);
    for ii=1:20
        yy = y(Fs*(ii*5):Fs*(ii*5 + 5));
        full_data(:, counter*20 + ii) = reshape(abs(spectrogram(yy)), [32769*8 1])';
    end
    counter = counter + 1;
end
X = full_data(:, [split_data 100 + split_data 200 + split_data]);
X2 = [X(:, 51:100) X(:, 151:200) X(:, 251:300)];
X = [X(:, 1:50) X(:, 101:150) X(:, 201:250)];
%% Test 1 Band Classification
% Naruto, Polo G, Avicii
% playblocking(audioplayer(yy,Fs))
an = 50; nn = 50; rn = 50;
feature=42; 
[U,S,V] = svd(X,'econ');
songs = S*V';
house = songs(1:feature,1:an);
hyperock = songs(1:feature,an+1:an+nn);
hiphop = songs(1:feature,an+nn+1:an+nn+rn);

mu = mean(songs(1:feature,:));
ma = mean(house,2);
mn = mean(hyperock,2);
mr = mean(hiphop,2);

%% 
Sw=0;  % within class variances
for i=1:an
    Sw = Sw + (house(:,i)-ma)*(house(:,i)-ma)';
end
for i=1:nn
    Sw = Sw + (hyperock(:,i)-mn)*(hyperock(:,i)-mn)';
end
for i = 1:rn
    Sw = Sw + (hiphop(:,i)-mr)*(hiphop(:,i)-mr)';
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
vhouse = w'*house; vhyperock = w'*hyperock; vhiphop = w'*hiphop;
vhouse1 = w1'*house; vhyperock1 = w1'*hyperock; vhiphop1 = w1'*hiphop;
figure(1)
plot(vhouse, vhouse1, 'ro'); hold on;
plot(vhyperock, vhyperock1, 'bo');
plot(vhiphop, vhiphop1, 'go'); hold off;
title('Training')
legend("House", 'Utility Power Metal', 'Hip Hop','Location', 'best')
xlabel("Eigenvector 1")
ylabel("Eigenvector 2")
%%
results = [vhouse vhyperock vhiphop];
res2 = [vhouse1 vhyperock1 vhiphop1];
tabl = [results' res2'];
trueRes = [repelem("House", 50) repelem("Power Metal", 50) repelem("Hip Hop", 50)]';
%%
Mdl = fitcknn(tabl,trueRes,'NumNeighbors',45,'Standardize',1);
label = predict(Mdl, tabl);
labelz = repelem("",150);
for hh = 1:150
   labelz(hh) = label{hh};
end
c = sum(trueRes==labelz')/150;

%% TEST 
an2 = 50; nn2 = 50; rn2 = 50;
feature=42; 
[U2,S2,V11] = svd(X2,'econ');
songs2 = S2*V11';
house2 = songs2(1:feature,1:an2);
hyperock2 = songs2(1:feature,an2+1:an2+nn2);
hiphop2 = songs2(1:feature,an2+nn2+1:an2+nn2+rn2);

mu2 = mean(songs2(1:feature,:));
ma2 = mean(house2,2);
mn2 = mean(hyperock2,2);
mr2 = mean(hiphop2,2);

%% 
Sw2=0;  % within class variances
for i=1:an2
    Sw2 = Sw2 + (house2(:,i)-ma2)*(house2(:,i)-ma2)';
end
for i=1:nn2
    Sw2 = Sw2 + (hyperock2(:,i)-mn2)*(hyperock2(:,i)-mn2)';
end
for i = 1:rn2
    Sw2 = Sw2 + (hiphop2(:,i)-mr2)*(hiphop2(:,i)-mr2)';
end
Sb2 = (ma2-mu2)*(ma2-mu2)' + (mn2-mu2)*(mn2-mu2)' + (mr2-mu2)*(mr2-mu2)';
[V22,D2] = eig(Sb2,Sw2);
[~,ind2] = max(abs(diag(D2)));
w2=V22(:,ind2); w2=w2/norm(w2,2);
%%
D332 = D2;
D332(:,ind2) = [];
D332(ind2, :) = [];
[~, indx2] = max(abs(diag(D332)));
w12 = V22(:, indx2 + 1); w12 = w12/norm(w12,2);
%%
vhouse2 = w2'*house2; vhyperock2 = w2'*hyperock2; vhiphop2 = w2'*hiphop2;
vhouse12 = w12'*house2; vhyperock12 = w12'*hyperock2; vhiphop12 = w12'*hiphop2;
results2 = [vhouse2 vhyperock2 vhiphop2];
res22 = [vhouse12 vhyperock12 vhiphop12];
tabl2 = [results2' res22'];
trueRes2 = [repelem("House", 50) repelem("Power Metal", 50) repelem("Hip Hop", 50)]';
%%
label2 = predict(Mdl, tabl2);
labelz2 = repelem("",150);
for hh = 1:150
   labelz2(hh) = label2{hh};
end
c2 = sum(trueRes2==labelz2')/150;
%%
figure(2)
plot(vhouse2, vhouse12, 'ro'); hold on;
plot(vhyperock2, vhyperock12, 'bo');
plot(vhiphop2, vhiphop12, 'go'); hold off;
title('Testing')
legend("House", 'Utility Power Metal', 'Hip Hop','Location', 'best')
xlabel("Eigenvector 1")
ylabel("Eigenvector 2")
