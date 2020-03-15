close all; clc; clear;
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')
numFrames2 = size(vidFrames1_2,4);

%% Test 2
for k = 1 : numFrames2
    mov(k).cdata = vidFrames1_2(:,:,:,k);
    mov(k).colormap = [];
    mov2(k).cdata = vidFrames2_2(:,:,:,k);
    mov2(k).colormap = [];
    mov3(k).cdata = vidFrames3_2(:,:,:,k);
    mov3(k).colormap = [];
end

x2 = zeros(1, numFrames2); y2 = x2;
slider = [15 18 19 20 22 25 30];
for kk = 1:length(slider)
    x24 = zeros(1, numFrames2); y24 = x24;
    for jj = 1:numFrames2
        X = rgb2gray(frame2im(mov(jj)));
        X(:, [1:256 441:end]) = 0; % hand calculated X
        X([1:193 292:end], :) = 0; % hand calculated Y
        [mmY, mmX] = find(X > max(X(:)) - slider(kk));
        y24(jj) = mean(mmY); 
        x24(jj) = mean(mmX);
    end
    y2 = y2 + y24;
    x2 = x2 + x24;
end
y2 = y2 ./ length(slider);
x2 = x2 ./ length(slider);

x21 = zeros(1, numFrames2); y21 = x21;
for kk = 1:length(slider)
    x24 = zeros(1, numFrames2); y24 = x24;
    for jj = 1:numFrames2
        X = rgb2gray(frame2im(mov2(jj)));
        X(:, [1:248 481:end]) = 0; % hand calculated X
        X([1:76 362:end], :) = 0; % hand calculated Y
        [mmY, mmX] = find(X > max(X(:)) - slider(kk));
        y24(jj) = mean(mmY); 
        x24(jj) = mean(mmX);
    end
    y21 = y21 + y24;
    x21 = x21 + x24;
end

y21 = y21 ./ length(slider);
x21 = x21 ./ length(slider);

x22 = zeros(1, numFrames2); y22 = x22;
for kk = 1:length(slider)
    x24 = zeros(1, numFrames2); y24 = x24;
    for jj = 1:numFrames2
        X = rgb2gray(frame2im(mov3(jj)));
        X(:, [1:270 452:end]) = 0; % hand calculated X
        X([1:172 312:end], :) = 0; % hand calculated Y
        [mmY, mmX] = find(X > max(X(:)) - slider(kk));
        y24(jj) = mean(mmY); 
        x24(jj) = mean(mmX);
    end
    y22 = y22 + y24;
    x22 = x22 + x24;
end
y22 = y22 ./ length(slider);
x22 = x22 ./ length(slider);

[~, ind] = min(y2(1:50)); 
x2 = x2(ind:end);
y2 = y2(ind:end);
[~, ind2] = min(y21(1:50));
x21 = x21(ind2:end);
y21 = y21(ind2:end);
[~, ind3] = min(x22(1:10));
x22 = x22(ind3:end);
y22 = y22(ind3:end);
y1 = y2; y11 = y21; y12 = y22;
x1 = x2; x11 = x21; x12 = x22;

%%
sh = min([length(x1), length(x11), length(x12)]);
x1 = x1(1:end - (length(x1) - sh));  y1 = y1(1:end - (length(y1) - sh));
x11 = x11(1:end - (length(x11) - sh)); y11 = y11(1:end - (length(y11) - sh));
x12 = x12(1:end - (length(x12) - sh)); y12 = y12(1:end - (length(y12) - sh));

%% FFT for smoothing
n = length(y1);
L = length(y1);
k = (2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1];
tau = 5;
k0 = 0;
filter = exp(-tau*(k-k0).^2); 
y1 = abs(ifft(filter.*fft(y1)));
y11 = abs(ifft(filter.*fft(y11)));
y12 = abs(ifft(filter.*fft(y12)));
x1 = abs(ifft(filter.*fft(x1)));
x11 = abs(ifft(filter.*fft(x11)));
x12 = abs(ifft(filter.*fft(x12)));

%%
figure(1)
subplot(3,2,1)
plot(y1)
xlabel('Time [Frame]')
ylabel('Position [Y]')
set(gca,'Fontsize',8) 
subplot(3,2,2)
plot(x1)
ylabel('Position [X]')
xlabel('Time [Frame]')
set(gca,'Fontsize',8) 
subplot(3,2,3)
plot(y11)
ylabel('Position [Y]')
xlabel('Time [Frame]')
set(gca,'Fontsize',8) 
subplot(3,2,4)
plot( x11)
ylabel('Position [X]')
xlabel('Time [Frame]')
set(gca,'Fontsize',8) 
subplot(3,2,5)
plot(y12)
ylabel('Position [Y]')
xlabel('Time [Frame]')
set(gca,'Fontsize',8) 
subplot(3,2,6)
plot(x12)
ylabel('Position [X]')
xlabel('Time [Frame]')
set(gca,'Fontsize',8) 
%% SVD
XMAT = [x1; y1; x11; y11; x12; y12];
[m,n]=size(XMAT);   %  compute data size
mn=mean(XMAT,2); %  compute mean for each row
XMAT=XMAT-repmat(mn,1,n);
X = XMAT;

[U,S,V]=svd(X/sqrt(n-1));  % perform the SVD
lambda=diag(S).^2;  % produce diagonal variances
Y = U'*X;% produce the principal components projection
%%
figure(2)
plot(lambda./sum(lambda), 'go', 'MarkerSize', 8);
hold on;
plot(lambda./sum(lambda), 'g--', 'Linewidth',2);
hold off;
set(gca, 'Fontsize', 8, 'Xtick',0:6)
title('Energy by Rank-N Approximation')
xlabel('Rank-N Approximation')
ylabel('Energy')

figure(3)
plot(Y(1,:)); hold on;
plot(Y(2,:)); 
plot(Y(3,:));
plot(Y(4,:));
plot(Y(5,:));
plot(Y(6,:)); hold off;
set(gca, 'Fontsize', 8)
title('Principal Components [PC]')
legend('PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', ...
    'PC 6', 'Location', 'best')
xlabel('Time [Frame]')
ylabel('Position')