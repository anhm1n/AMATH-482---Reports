close all; clear; clc;
load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')

numFrames4 = size(vidFrames1_4,4);

%% Test 4
for k = 1 : numFrames4
    mov(k).cdata = vidFrames1_4(:,:,:,k);
    mov(k).colormap = [];
    mov2(k).cdata = vidFrames2_4(:,:,:,k);
    mov2(k).colormap = [];
    mov3(k).cdata = vidFrames3_4(:,:,:,k);
    mov3(k).colormap = [];
end
x4 = zeros(1, numFrames4); y4 = x4;
for jj = 1:numFrames4
   X = rgb2gray(frame2im(mov(jj)));
   X(:, [1:320 450:end]) = 0; % hand calculated X
   X([1:230 385:end], :) = 0; % hand calculated Y
   [mmY, mmX] = find(X > max(X(:)) - 20);
   y4(jj) = mean(mmY);
   x4(jj) = mean(mmX);
end

x41 = zeros(1, numFrames4); y41 = x41;
for jj = 1:numFrames4
   X = rgb2gray(frame2im(mov2(jj)));
   X(:, [1:197 380:end]) = 0; % hand calculated X
   X([1:75 356:end], :) = 0; % hand calculated Y
   [mmY, mmX] = find(X > max(X(:)) - 20);
   y41(jj) = mean(mmY);
   x41(jj) = mean(mmX);
end

x42 = zeros(1, numFrames4); y42 = x42;
for jj = 1:numFrames4
   X = rgb2gray(frame2im(mov3(jj)));
   X(:, [1:271 495:end]) = 0; % hand calculated X
   X([1:173 405:end], :) = 0; % hand calculated Y
   [mmY, mmX] = find(X > max(X(:)) - 20);
   y42(jj) = mean(mmY);
   x42(jj) = mean(mmX);
end

[~, ind] = min(y4(1:20)); 
x4 = x4(ind:end);
y4 = y4(ind:end);
[~, ind2] = min(y41(1:25));
x41 = x41(ind2:end);
y41 = y41(ind2:end);
[~, ind3] = min(x42(1:20));
x42 = x42(ind3:end);
y42 = y42(ind3:end);

y1 = y4; y11 = y41; y12 = y42;
x1 = x4; x11 = x41; x12 = x42;

%%
sh = min([length(x1), length(x11), length(x12)]);
x1 = x1(1:end - (length(x1) - sh));  y1 = y1(1:end - (length(y1) - sh));
x11 = x11(1:end - (length(x11) - sh)); y11 = y11(1:end - (length(y11) - sh));
x12 = x12(1:end - (length(x12) - sh)); y12 = y12(1:end - (length(y12) - sh));

%% FFT for smoothing
n = length(y1);
L = length(y1);
k = (2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1];
tau = .5;
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