%% EXTRACT SPECTRAL CENTROID, FLUX, ZCR, AND PITCH
% For ASVspoof 2019 LA dataset - SIMPLE ROBUST PITCH
clear; clc; close all;

%% SET YOUR AUDIO PATH
audioParentPath = 'E:\LA\LA';

%% Load existing MFCC data
load('ASVspoof_features.mat');  % Loads X (MFCCs), Y (labels)
fprintf('Loaded %d files from MFCC data\n', size(X,1));

%% Find ALL FLAC files (they are in 'flac' subfolders)
trainFiles = dir(fullfile(audioParentPath, 'ASVspoof2019_LA_train', 'flac', '*.flac'));
devFiles = dir(fullfile(audioParentPath, 'ASVspoof2019_LA_dev', 'flac', '*.flac'));
evalFiles = dir(fullfile(audioParentPath, 'ASVspoof2019_LA_eval', 'flac', '*.flac'));

fprintf('\nFound audio files:\n');
fprintf('  Train: %d files\n', length(trainFiles));
fprintf('  Dev:   %d files\n', length(devFiles));
fprintf('  Eval:  %d files\n', length(evalFiles));

% Combine all file info (ONLY the train files)
allFiles = [];

for i = 1:length(trainFiles)
    allFiles(end+1).name = trainFiles(i).name;
    allFiles(end).folder = fullfile(audioParentPath, 'ASVspoof2019_LA_train', 'flac');
    allFiles(end).set = 'train';
end

numAudioFiles = length(allFiles);
fprintf('\nUsing only TRAIN files: %d files\n', numAudioFiles);
fprintf('MFCC data has: %d files\n', size(X,1));

% Process ALL matching files
numToProcess = min(numAudioFiles, size(X,1));
fprintf('\nWill process %d files\n', numToProcess);

%% Pre-allocate for features
centroid_mean = zeros(numToProcess, 1);
centroid_std = zeros(numToProcess, 1);
flux_mean = zeros(numToProcess, 1);
flux_std = zeros(numToProcess, 1);
zcr_mean = zeros(numToProcess, 1);
zcr_std = zeros(numToProcess, 1);
pitch_mean = zeros(numToProcess, 1);
pitch_std = zeros(numToProcess, 1);

%% Process each file
fprintf('\nStarting feature extraction...\n');
tic;

for fileIdx = 1:numToProcess
    % Show progress every 500 files
    if mod(fileIdx, 500) == 0 || fileIdx == 1
        fprintf('Processing %d of %d... (%.1f%%)\n', fileIdx, numToProcess, 100*fileIdx/numToProcess);
    end
    
    % Load audio
    audioFile = fullfile(allFiles(fileIdx).folder, allFiles(fileIdx).name);
    
    try
        [audio, fs] = audioread(audioFile);
    catch ME
        fprintf('  ERROR: Could not load %s\n', allFiles(fileIdx).name);
        continue;
    end
    
    % Convert to mono if needed
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end
    
    % Frame parameters (25ms frames, 10ms hop at 16kHz)
    frameLen = round(0.025 * fs);  % 400 samples
    hopLen = round(0.010 * fs);    % 160 samples
    
    numFrames = floor((length(audio) - frameLen) / hopLen) + 1;
    if numFrames < 5
        continue;
    end
    
    % Pre-allocate
    centroid_frames = zeros(numFrames, 1);
    zcr_frames = zeros(numFrames, 1);
    pitch_frames = zeros(numFrames, 1);
    flux_frames = zeros(numFrames-1, 1);
    
    % Frequency axis for centroid
    freq = linspace(0, fs/2, frameLen/2 + 1);
    
    prev_mag = [];
    
    for i = 1:numFrames
        % Extract frame with window
        startIdx = (i-1)*hopLen + 1;
        frame = audio(startIdx:startIdx+frameLen-1);
        frame = frame .* hamming(frameLen);
        
        % FFT
        Y = abs(fft(frame));
        Y = Y(1:frameLen/2 + 1);
        
        % 1. SPECTRAL CENTROID
        if sum(Y) > 0
            centroid_frames(i) = sum(freq .* Y') / sum(Y);
            centroid_frames(i) = centroid_frames(1);  % Ensure scalar
        else
            centroid_frames(i) = 0;
        end
        
        % 2. ZERO CROSSING RATE
        zcr_frames(i) = sum(abs(diff(sign(frame)))) / (2*frameLen);
        
        % 3. PITCH - SIMPLE AUTOCORRELATION (NO findpeaks)
        [r, lags] = xcorr(frame, 'normalized');
        r = r(lags >= 0);
        lags = lags(lags >= 0);
        
        % Find the maximum peak after zero-lag (index 1)
        % Search from index 2 to end (skip zero-lag)
        [max_val, max_idx_rel] = max(r(2:end));
        max_idx = max_idx_rel + 1;  % +1 because we skipped index 1
        
        % Check if peak is significant
        if max_val > 0.3  % Threshold
            pitch_candidate = fs / lags(max_idx);
            % Human speech range: 80-300 Hz
            if pitch_candidate >= 70 && pitch_candidate <= 350
                pitch_frames(i) = pitch_candidate;
            else
                pitch_frames(i) = 0;
            end
        else
            pitch_frames(i) = 0;
        end
        
        % 4. SPECTRAL FLUX
        if i > 1 && ~isempty(prev_mag)
            curr_norm = Y / (max(Y) + eps);
            flux_frames(i-1) = sqrt(sum((curr_norm - prev_mag).^2));
        end
        prev_mag = Y / (max(Y) + eps);
    end
    
    % Store statistics
    centroid_mean(fileIdx) = mean(centroid_frames);
    centroid_std(fileIdx) = std(centroid_frames);
    flux_mean(fileIdx) = mean(flux_frames);
    flux_std(fileIdx) = std(flux_frames);
    zcr_mean(fileIdx) = mean(zcr_frames);
    zcr_std(fileIdx) = std(zcr_frames);
    
    pitch_nonzero = pitch_frames(pitch_frames > 0);
    if isempty(pitch_nonzero)
        pitch_mean(fileIdx) = 0;
        pitch_std(fileIdx) = 0;
    else
        pitch_mean(fileIdx) = mean(pitch_nonzero);
        pitch_std(fileIdx) = std(pitch_nonzero);
    end
end

elapsed = toc;
fprintf('\nCompleted in %.2f minutes\n', elapsed/60);

%% Save features
myFeatures = [centroid_mean, centroid_std, flux_mean, flux_std, ...
              zcr_mean, zcr_std, pitch_mean, pitch_std];

save('my_audio_features.mat', 'myFeatures', ...
     'centroid_mean', 'centroid_std', 'flux_mean', 'flux_std', ...
     'zcr_mean', 'zcr_std', 'pitch_mean', 'pitch_std');

fprintf('\nSaved my_audio_features.mat\n');

%% Summary
fprintf('\n========== SUMMARY ==========\n');
fprintf('Files processed: %d\n', numToProcess);
fprintf('\nAverage values across all processed files:\n');
fprintf('  Spectral Centroid: %.2f Hz (std: %.2f)\n', mean(centroid_mean), mean(centroid_std));
fprintf('  Spectral Flux:     %.4f (std: %.4f)\n', mean(flux_mean), mean(flux_std));
fprintf('  Zero Crossing Rate: %.4f (std: %.4f)\n', mean(zcr_mean), mean(zcr_std));
fprintf('  Pitch:             %.2f Hz (std: %.2f)\n', mean(pitch_mean), mean(pitch_std));

%% Histograms of features
figure('Position', [100, 100, 1200, 400]);

subplot(1,4,1);
histogram(centroid_mean, 50);
xlabel('Frequency (Hz)'); ylabel('Count');
title('Spectral Centroid');
grid on;

subplot(1,4,2);
histogram(flux_mean, 50);
xlabel('Flux'); ylabel('Count');
title('Spectral Flux');
grid on;

subplot(1,4,3);
histogram(zcr_mean, 50);
xlabel('ZCR'); ylabel('Count');
title('Zero Crossing Rate');
grid on;

subplot(1,4,4);
histogram(pitch_mean(pitch_mean > 0), 50);
xlabel('Frequency (Hz)'); ylabel('Count');
title('Pitch (non-zero)');
grid on;

sgtitle('Distribution of Extracted Features (All Files)');