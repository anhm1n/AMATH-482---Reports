%% Piano
close all; clc; clear all;
figure(8)
[y,Fs] = audioread('music1.wav');
tr_piano=length(y)/Fs;  % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');
% p8 = audioplayer(y,Fs); playblocking(p8);

%% Recorder
figure(9)
[y2,Fs2] = audioread('music2.wav');
tr_rec=length(y2)/Fs2;  % record time in seconds
%plot((1:length(y2))/Fs2,y2);
%xlabel('Time [sec]'); ylabel('Amplitude');
%title('Mary had a little lamb (recorder)');
% p8 = audioplayer(y2,Fs2); playblocking(p8)

%% Fourier Transform 
v = y'; v2 = y2';
n = length(v) ; n2 = length(v2);
t = (1:length(v))/Fs; t2 = (1:length(v2))/Fs2;
k = (2*pi/tr_piano)*[0:n/2-1 -n/2:-1];
k2 = (2*pi/tr_rec)*[0:n2/2-1 -n2/2:-1];
ks=fftshift(k); ks2=fftshift(k2);
%% Gabor Transformation Piano %%
close all;
a = 40;
tslide=0:.21:tr_piano;
hertz_maxes = tslide * 0;
vgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg);
    [M, Ind] = max(abs(vgt));
    hertz_maxes(j) = abs(k(Ind)/(2*pi));
    vgt_spec(j,:) = fftshift(abs(vgt))/max(abs(vgt)); 
    
end

%% Spectrogram Piano %%
figure(10)
pcolor(tslide, ks/(2*pi),vgt_spec.'), 
shading interp 
title(['Piano Spectrogram a = ',num2str(a)],'Fontsize',8)
set(gca,'Fontsize',8, 'Ylim', [0 500]) 
xlabel('Time [sec]');
ylabel('Frequency [Hertz]');
colormap(winter)
colorbar
%% Gabor Transformation Recorder %%
close all;
a = 50;
tslide2=0:.2:tr_rec;
vgt_spec2 = zeros(length(tslide2),n2);
hertz_maxes2 = tslide2*0;
for j=1:length(tslide2)
    g=exp(-a*(t2-tslide2(j)).^2); 
    vg=g.*v2; 
    vgt=fft(vg); 
    [M, ind] = max(abs(vgt));
    hertz_maxes2(j) = abs(k(ind)/(2*pi));
    vgt_spec2(j,:) = fftshift(abs(vgt))/max(abs(vgt)); 
end

%% Spectrogram Recorder %%
figure(11)
pcolor(tslide2, ks2/(2*pi), vgt_spec2.'),
shading interp 
title(['Recorder Spectrogram a = ',num2str(a)],'Fontsize',8)
set(gca,'Fontsize',8, "Ylim", [600 1300]) 
xlabel('Time [sec]');
ylabel('Frequency [Hertz]');
colormap(winter)
colorbar

%% Songs
figure(12)
subplot(1, 2, 1)
plot(tslide, hertz_maxes)
title("Piano Central Frequencies")
xlabel("Time [sec]")
ylabel('Frequency [Hertz]')

subplot(1, 2, 2)
plot(tslide2, hertz_maxes2)
title("Recorder Central Frequencies")
xlabel("Time [sec]")
ylabel('Frequency [Hertz]')
