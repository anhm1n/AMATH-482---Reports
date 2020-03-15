clear; close all; clc;
load Testdata
L=15; % spatial domain
n=64; % Fourier modes
leng=262144;
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);
%% Average together 20 noisy signals in the Fourier domain
m = 20;
ave = zeros(n, n, n);
UndataFT = zeros(m,n,n,n);

for j=1:m
    UndataFT(j,:,:,:) = fftn(reshape(Undata(j,:),n,n,n));
    ave = ave + squeeze(UndataFT(j,:,:,:));
end
aveft = abs(fftshift(ave))/m;
aveft = abs(aveft./max(abs(aveft), [], 'all'));

%% Graph the averaged frequencies
close all; clc;
figure(1);
isosurface(Kx,Ky,Kz, aveft,0.6)
axis([-20 20 -20 20 -20 20]), grid on
%% Find the index of max value from averaging & apply filter 
[mm, index] = max(aveft(:));
[px,py,pz] = ind2sub([n,n,n], index); % 28, 42, 33

tau = 0.2;
k0x = Kx(px,py,pz); % 1.8850
k0y = Ky(px,py,pz); % -1.0472
k0z = Kz(px,py,pz); % 0
filter = exp((-tau*(Kx-k0x).^2) + (-tau*(Ky-k0y).^2) + (-tau*(Kz-k0z).^2));
filter = fftshift(filter); % Recall that our values need to be shifted
marbleTrajectory = zeros(m,3);
for j=1:m
    UndataS = ifftn(filter.*squeeze(UndataFT(j,:,:,:)));
    [mmm, indx] = max(UndataS(:));
    [mx, my, mz] = ind2sub([n,n,n], indx);
    marbleTrajectory(j,1) = X(mx,my,mz);
    marbleTrajectory(j,2) = Y(mx,my,mz);
    marbleTrajectory(j,3) = Z(mx,my,mz);
end

%% Graph trajectory of marble
clf; close all;
figure(2)
plot3(marbleTrajectory(:,1), marbleTrajectory(:,2), marbleTrajectory(:,3), 'LineWidth', 2) 
hold on
plot3(marbleTrajectory(:,1), marbleTrajectory(:,2), marbleTrajectory(:,3), 'r*')
plot3(marbleTrajectory(20,1), marbleTrajectory(20,2), marbleTrajectory(20,3), 'db', 'MarkerSize', 12)
% -5.6250    4.2188   -6.0938
axis([-12 12 -12 12 -12 12]), grid on
hold off; 