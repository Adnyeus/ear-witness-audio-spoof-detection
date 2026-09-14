%% ============================================================
%  ASVspoof 2019 LA - Audio Loading Pipeline (Efficient Version)
%  This script:
%  1. Reads protocol file (labels)
%  2. Iterates through ALL audio files
%  3. Loads one file at a time (memory-safe)
%  4. Stores useful metadata
% =============================================================

clc;
clear;
close all;

%% ===================== STEP 1: SET PATHS =====================

% Path to folder containing .flac audio files
audioPath = 'C:\Users\Khadija\Documents\6thSemester\Digital Signal P\Project\LA\ASVspoof2019_LA_train\flac\';

% Path to protocol file (labels)
protocolPath = 'C:\Users\Khadija\Documents\6thSemester\Digital Signal P\Project\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt';

%% ===================== STEP 2: READ PROTOCOL =================

% Open the protocol file
fid = fopen(protocolPath);

% Read columns:
% Column 1 → Speaker ID
% Column 2 → Audio File Name
% Column 3 → '-'
% Column 4 → System ID
% Column 5 → Label (bonafide / spoof)
data = textscan(fid, '%s %s %s %s %s');

% Close file after reading
fclose(fid);

% Extract relevant columns
fileNames = data{2};   % Audio file names (without .flac)
labels = data{5};      % Corresponding labels

% Total number of files
numFiles = length(fileNames);

disp(['Total files to process: ', num2str(numFiles)]);
%% ===================== STEP 3: INITIALIZE STORAGE ============

% Pre-allocate arrays for efficiency

durations = zeros(numFiles,1);       % Duration of each audio
samplingRates = zeros(numFiles,1);   % Sampling frequency
labelArray = strings(numFiles,1);    % Labels (bonafide/spoof)

% NOTE:
% We DO NOT store raw audio → too memory heavy
% Later we will store features instead

%% ===================== STEP 4: PROCESS ALL FILES =============

for i = 1:numFiles
    
    % -------- Construct full file path --------
    filePath = fullfile(audioPath, [fileNames{i}, '.flac']);
    
    % -------- Check if file exists --------
    if ~isfile(filePath)
        warning(['File not found: ', filePath]);
        continue;
    end
    
    % -------- Load audio --------
    % audio → signal values
    % fs → sampling frequency (should be 16000 Hz)
    [audio, fs] = audioread(filePath);
    
    % -------- Convert to mono if stereo --------
    if size(audio,2) > 1
        audio = mean(audio, 2);
    end
    
    % -------- Store metadata --------
    durations(i) = length(audio) / fs;   % duration in seconds
    samplingRates(i) = fs;               % sampling rate
    labelArray(i) = labels{i};           % bonafide / spoof
    
    % -------- Progress display --------
    if mod(i,500) == 0
        disp(['Processed ', num2str(i), ' / ', num2str(numFiles), ' files']);
    end
    
end

%% ===================== STEP 5: DATASET SUMMARY ===============

disp('----- DATASET SUMMARY -----');

% Average duration of audio files
avgDuration = mean(durations);
disp(['Average Duration (seconds): ', num2str(avgDuration)]);

% Unique sampling rates (should be 16000 Hz)
uniqueFs = unique(samplingRates);
disp('Unique Sampling Rates:');
disp(uniqueFs);

% Count bonafide and spoof samples
numBonafide = sum(labelArray == "bonafide");
numSpoof = sum(labelArray == "spoof");

disp(['Number of Bonafide samples: ', num2str(numBonafide)]);
disp(['Number of Spoof samples: ', num2str(numSpoof)]);

%% ===================== STEP 6: SAVE RESULTS ==================

% Save processed metadata for later use
save('audio_metadata.mat', 'durations', 'samplingRates', 'labelArray');

disp('Metadata saved successfully.');

%% ===================== END OF SCRIPT =========================