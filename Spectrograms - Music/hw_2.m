clear; close all; clc;
load handel
v = y';
v = v(1:length(v));

figure(1)
plot((1:length(v))/Fs,v);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');

%% Fourier Transform 
L = length(y)/Fs; n = length(v);
t = (1:length(v))/Fs;
k = (2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1];
ks=fftshift(k);
vt = fft(v);
%% FFT
figure(2)
tslide=0:0.1:L;
vgt_spec = repmat(fftshift(abs(vt)),length(tslide),1);
pcolor(tslide,ks,vgt_spec.'), 
shading interp 
title('fft','Fontsize',8)
xlabel("Time (sec)");
ylabel("Frequency (\omega)");
set(gca,'Fontsize',8) 
colormap(hot) 
colorbar

%% Gabor Construction t = 0.1%%
figure(3)
a_vec = [400 200 100 1];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:L;
    vgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        vg=g.*v; 
        vgt=fft(vg); 
        vgt_spec(j,:) = fftshift(abs(vgt)); 
    end
    subplot(2,2,jj)
    pcolor(tslide,ks,vgt_spec.'), 
    shading interp 
    title(['a = ',num2str(a)],'Fontsize',8)
    set(gca,'Fontsize',8, 'Ylim', [0 10000]) 
    xlabel("Time [sec]");
    ylabel('Frequency [\omega]');
    colormap(hot) 
end

%% Gabor Construction t = 0.5 %%
figure(4)
a_vec = [400 200 100 1];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.5:L;
    vgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        vg=g.*v; 
        vgt=fft(vg); 
        vgt_spec(j,:) = fftshift(abs(vgt)); 
    end
    subplot(2,2,jj)
    pcolor(tslide,ks,vgt_spec.'), 
    shading interp 
    title(['a = ',num2str(a)],'Fontsize',8)
    set(gca,'Fontsize',8, 'Ylim', [0 10000])
    xlabel("Time [sec]");
    ylabel('Frequency [\omega]');
    colormap(hot) 
end

%% Mexican Hat %%
figure(5)
a_vec = [1 .25 .1 .01];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:L;
    vgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        mexicanhat = 2 / (sqrt(3*a)*(pi)^(1/4)) * ...
        (1-((t-tslide(j))/a).^2).* ...
        exp(-(t-tslide(j)).^2 / (2*a^2)); 
        vg=mexicanhat.*v; 
        vgt=fft(vg); 
        vgt_spec(j,:) = fftshift(abs(vgt)); 
    end
    subplot(2,2,jj)
    pcolor(tslide,ks,vgt_spec.'), 
    shading interp 
    title(['Mexican Hat a = ',num2str(a)],'Fontsize',8)
    set(gca,'Fontsize',8, 'Ylim', [0 12000]) 
    xlabel("Time [sec]");
    ylabel('Frequency [\omega]');
    colormap(hot) 
end

%% Shannon %%
figure(6)
a_vec = [1 .25 .1 .01];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:9;
    vgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        shannon = (abs(t-tslide(j)) < a);
        vg=shannon.*v; 
        vgt=fft(vg); 
        vgt_spec(j,:) = fftshift(abs(vgt)); 
    end
    subplot(2,2,jj)
    pcolor(tslide,ks,vgt_spec.'), 
    shading interp 
    title(['Shannon a = ',num2str(a)],'Fontsize',8)
    set(gca,'Fontsize',8, 'Ylim', [0 12000]) 
    xlabel("Time [sec]");
    ylabel('Frequency [\omega]');
    colormap(hot) 
end

%% Gabor Comparison %%
figure(7)
a_vec = [1 .25 .1 .01];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.5:L;
    vgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        vg=g.*v; 
        vgt=fft(vg); 
        vgt_spec(j,:) = fftshift(abs(vgt)); 
    end
    subplot(2,2,jj)
    pcolor(tslide,ks,vgt_spec.'), 
    shading interp 
    title(['a = ',num2str(a)],'Fontsize',8)
    set(gca,'Fontsize',8, 'Ylim', [0 10000]) 
    xlabel("Time [sec]");
    ylabel('Frequency [\omega]');
    colormap(hot) 
end