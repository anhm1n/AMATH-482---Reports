clear; clc; close all;
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')

numFrames3 = size(vidFrames3_3,4);

%% Test 3
for k = 1 : numFrames3
    mov(k).cdata = vidFrames1_3(:,:,:,k);
    mov(k).colormap = [];
    mov2(k).cdata = vidFrames2_3(:,:,:,k);
    mov2(k).colormap = [];
    mov3(k).cdata = vidFrames3_3(:,:,:,k);
    mov3(k).colormap = [];
end
x3 = zeros(1, numFrames3); y3 = x3; 
for jj = 1:numFrames3
   X = rgb2gray(frame2im(mov(jj)));
   X(:, [1:260 400:end]) = 0; % hand calculated X
   X([1:200 389:end], :) = 0; % hand calculated Y
   [mmY, mmX] = find(X > max(X(:)) - 30);
   y3(jj) = mean(mmY);
   x3(jj) = mean(mmX);
end

x31 = zeros(1, numFrames3); y31 = x31; 
for jj = 1:numFrames3
   X = rgb2gray(frame2im(mov2(jj)));
   X(:, [1:230 397:end]) = 0; % hand calculated X
   X([1:160 376:end], :) = 0; % hand calculated Y
   [mmY, mmX] = find(X > max(X(:)) - 30);
   y31(jj) = mean(mmY);
   x31(jj) = mean(mmX);
end

x32 = zeros(1, numFrames3); y32 = x32; 
for jj = 1:numFrames3
   X = rgb2gray(frame2im(mov3(jj)));
   X(:, [1:148 437:end]) = 0; % hand calculated X
   X([1:211 337:end], :) = 0; % hand calculated Y
   [mmY, mmX] = find(X > max(X(:)) - 30);
   y32(jj) = mean(mmY);
   x32(jj) = mean(mmX);
end

[~, ind] = max(y3(1:30)); 
x3 = x3(ind:end);
y3 = y3(ind:end);
[~, ind2] = max(y31(1:30));
x31 = x31(ind2:end);
y31 = y31(ind2:end);
[~, ind3] = max(x32(1:30));
x32 = x32(ind3:end);
y32 = y32(ind3:end);

y1 = y3; y11 = y31; y12 = y32;
x1 = x3; x11 = x31; x12 = x32;

%%
sh = min([length(x1), length(x11), length(x12)]);
x1 = x1(1:end - (length(x1) - sh));  y1 = y1(1:end - (length(y1) - sh));
x11 = x11(1:end - (length(x11) - sh)); y11 = y11(1:end - (length(y11) - sh));
x12 = x12(1:end - (length(x12) - sh)); y12 = y12(1:end - (length(y12) - sh));

%% FFT for smoothing
n = length(y1);
L = length(y1);
k = (2*pi/L)*[0:n/2-1 -n/2:-1];
tau = .2;
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