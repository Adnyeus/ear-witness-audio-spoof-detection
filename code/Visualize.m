%% ============================================================
%  ASVspoof 2019 - Audio Visualization Script
%  This script:
%  1. Loads metadata (labels, file names)
%  2. Selects a small subset (bonafide + spoof)
%  3. Loads selected audio files
%  4. Plots waveform and spectrogram
% =============================================================

clc;
clear;
close all;

%% ===================== STEP 1: SET PATH =====================

% Path to audio files
audioPath = 'C:\Users\Khadija\Documents\6thSemester\Digital Signal P\Project\LA\ASVspoof2019_LA_train\flac\';

% Load metadata (from previous script)
load('audio_metadata.mat');  
% This contains: labelArray, durations, samplingRates

% IMPORTANT:
% We also need fileNames again → reload protocol

protocolPath = 'C:\Users\Khadija\Documents\6thSemester\Digital Signal P\Project\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt';

fid = fopen(protocolPath);
data = textscan(fid, '%s %s %s %s %s');
fclose(fid);

fileNames = data{2};   % audio file names


%% ===================== STEP 2: FIND INDICES =================

% Find indices for each class
bonafide_idx = find(labelArray == "bonafide");
spoof_idx = find(labelArray == "spoof");

%% ===================== STEP 3: SELECT SAMPLES ===============

% Number of samples per class (keep small for clarity)
numSamples = 2;

% Random selection (better than fixed selection)
rng(1); % for reproducibility

selected_bonafide = bonafide_idx(randperm(length(bonafide_idx), numSamples));
selected_spoof = spoof_idx(randperm(length(spoof_idx), numSamples));

%% ===================== STEP 4: PLOTTING =====================

figure;

plotIndex = 1;

%% -------- BONAFIDE SAMPLES --------
for i = 1:length(selected_bonafide)
    
    idx = selected_bonafide(i);          % get index
    fileName = fileNames{idx};           % file name
    
    % Construct full path
    filePath = fullfile(audioPath, [fileName '.flac']);
    
    % Load audio
    [audio, fs] = audioread(filePath);
    
    % Convert to mono if stereo
    if size(audio,2) > 1
        audio = mean(audio,2);
    end
    
    % Create time axis
    t = (0:length(audio)-1)/fs;
    
    % -------- Plot waveform --------
    subplot(4,2,plotIndex);
    plot(t, audio);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['Bonafide Waveform: ', fileName]);
    grid on;
    
    plotIndex = plotIndex + 1;
    
    % -------- Plot spectrogram --------
    subplot(4,2,plotIndex);
    spectrogram(audio, 256, 200, 256, fs, 'yaxis');
    title(['Bonafide Spectrogram: ', fileName]);
    colorbar;
    
    plotIndex = plotIndex + 1;
end

%% -------- SPOOF SAMPLES --------
for i = 1:length(selected_spoof)
    
    idx = selected_spoof(i);
    fileName = fileNames{idx};
    
    filePath = fullfile(audioPath, [fileName '.flac']);
    
    [audio, fs] = audioread(filePath);
    
    if size(audio,2) > 1
        audio = mean(audio,2);
    end
    
    t = (0:length(audio)-1)/fs;
    
    % -------- Plot waveform --------
    subplot(4,2,plotIndex);
    plot(t, audio);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['Spoof Waveform: ', fileName]);
    grid on;
    
    plotIndex = plotIndex + 1;
    
    % -------- Plot spectrogram --------
    subplot(4,2,plotIndex);
    spectrogram(audio, 256, 200, 256, fs, 'yaxis');
    title(['Spoof Spectrogram: ', fileName]);
    colorbar;
    
    plotIndex = plotIndex + 1;
end

%% ===================== DONE =====================

disp('Visualization complete.');