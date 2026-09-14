clc;
clear;
close all;

%% ===================== STEP 1: SET PATHS =====================

audioPath = 'C:\Users\Khizer Kashif\OneDrive\OneDrive - National University of Sciences & Technology\projects\digital signal processing\Dataset\LA\ASVspoof2019_LA_train\flac';
protocolPath = 'C:\Users\Khizer Kashif\OneDrive\OneDrive - National University of Sciences & Technology\projects\digital signal processing\Dataset\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt';

%% ===================== STEP 2: READ PROTOCOL =================

fid = fopen(protocolPath);
data = textscan(fid, '%s %s %s %s %s');
fclose(fid);

fileNames = data{2};
labels = data{5};
numFiles = length(fileNames);

disp(['Total files to process: ', num2str(numFiles)]);

%% ===================== STEP 3: INITIALIZE STORAGE ============

durations = zeros(numFiles,1);
samplingRates = zeros(numFiles,1);
labelArray = strings(numFiles,1);

%% ===================== STEP 4: FEATURE STORAGE ===============

% 13 static + 13 delta + 13 delta-delta = 39 coefficients
% Mean and standard deviation for each → 39*2 = 78 features
numFeatures = 78;
X = zeros(numFiles, numFeatures);
Y = zeros(numFiles,1);

% Track which files are successfully processed
validFlags = false(numFiles,1);

%% ===================== STEP 5: PROCESS ALL FILES =============

for i = 1:numFiles
    
    filePath = fullfile(audioPath, [fileNames{i}, '.flac']);
    
    if ~isfile(filePath)
        warning(['File not found: ', filePath]);
        continue;
    end
    
    [audio, fs] = audioread(filePath);
    
    if size(audio,2) > 1
        audio = mean(audio, 2);
    end
    
    % ---- Skip very short files (less than 0.5 seconds) ----
    if length(audio)/fs < 0.5
        warning(['Skipping short file (<0.5s): ', fileNames{i}]);
        continue;
    end
    
    % ---- Pre-emphasis (typical filter to boost high frequencies) ----
    audio = filter([1 -0.97], 1, audio);
    
    % ---- Peak normalization (prevents MFCC issues) ----
    audio = audio ./ max(abs(audio)+1e-6);
    
    %% ===== ROBUST MFCC + DELTA + DELTA-DELTA =====
    coeffs = mfcc(audio, fs, 'NumCoeffs', 13);
    
    % Force 13 coefficients (some versions may include 0‑th coeff)
    coeffs = coeffs(:, 1:min(13, size(coeffs,2)));
    
    % Compute delta and delta-delta (uses default odd window length = 9)
    delta   = audioDelta(coeffs);    % FIXED: no explicit window → default odd (9)
    delta2  = audioDelta(delta);
    
    % Transpose to [coefficient × frames] for statistics (13 rows each)
    static  = coeffs';
    delta   = delta';
    delta2  = delta2';
    
    % Mean and standard deviation across frames (each column is a coefficient)
    mfcc_mean = [mean(static,2); mean(delta,2); mean(delta2,2)];   % 39×1
    mfcc_std  = [std(static,0,2); std(delta,0,2); std(delta2,0,2)]; % 39×1
    
    featureVector = [mfcc_mean; mfcc_std]';   % 1 × 78
    
    X(i,:) = featureVector;
    
    %% -------- Store metadata --------
    durations(i) = length(audio) / fs;
    samplingRates(i) = fs;
    labelArray(i) = labels{i};
    
    if labels{i} == "bonafide"
        Y(i) = 0;
    else
        Y(i) = 1;
    end
    
    validFlags(i) = true;   % Mark this row as processed
    
    if mod(i,500) == 0
        disp(['Processed ', num2str(i), ' / ', num2str(numFiles), ' files']);
    end
    
end

%% ===================== REMOVE UNPROCESSED ROWS ===============
% (Missing files or skipped short files left zero rows – drop them)
X = X(validFlags, :);
Y = Y(validFlags);
labelArray = labelArray(validFlags);
durations = durations(validFlags);
samplingRates = samplingRates(validFlags);

%% ===================== STEP 6: DATASET SUMMARY ===============

disp('----- DATASET SUMMARY -----');

avgDuration = mean(durations);
disp(['Average Duration (seconds): ', num2str(avgDuration)]);

uniqueFs = unique(samplingRates);
disp('Unique Sampling Rates:');
disp(uniqueFs);

numBonafide = sum(labelArray == "bonafide");
numSpoof = sum(labelArray == "spoof");

disp(['Number of Bonafide samples: ', num2str(numBonafide)]);
disp(['Number of Spoof samples: ', num2str(numSpoof)]);

%% ===================== SAVE FEATURES ==================

save('ASVspoof_features.mat', 'X', 'Y', 'labelArray', 'durations', 'samplingRates');
disp('Features extracted and saved successfully!');

%% ===================== END OF SCRIPT =======================