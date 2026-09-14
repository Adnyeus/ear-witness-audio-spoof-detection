%% COMBINE FEATURES FROM KHIZER AND FARHAN
% Script by: Ebad
% Description: Combines MFCC features (78) with spectral features (8)
%              to create a single feature matrix (86 features total)

clear; clc; close all;

%% Step 1: Load both feature files
fprintf('========================================\n');
fprintf('STEP 1: Loading feature files\n');
fprintf('========================================\n');

% Load Khizer's MFCC features
fprintf('Loading MFCC features from Khizer...\n');
load('ASVspoof_features.mat');
% Variables loaded: X (25380x78), Y (25380x1), durations, samplingRates, labelArray

% Load Farhan's spectral features
fprintf('Loading spectral features from Farhan...\n');
load('my_audio_features.mat');
% Variables loaded: myFeatures (25380x8), centroid_mean, centroid_std, etc.

%% Step 2: Verify dimensions match
fprintf('\n========================================\n');
fprintf('STEP 2: Verifying dimensions\n');
fprintf('========================================\n');

fprintf('Khizer MFCC features:    %d files × %d features\n', size(X,1), size(X,2));
fprintf('Farhan spectral features: %d files × %d features\n', size(myFeatures,1), size(myFeatures,2));

if size(X,1) ~= size(myFeatures,1)
    error('ERROR: Number of files does not match! Cannot combine.');
else
    fprintf('✓ File counts match: %d files\n', size(X,1));
end

%% Step 3: Combine features side-by-side
fprintf('\n========================================\n');
fprintf('STEP 3: Combining features\n');
fprintf('========================================\n');

% Concatenate horizontally
X_combined = [X, myFeatures];  % 25380 × 86

fprintf('Combined feature matrix size: %d files × %d features\n', ...
    size(X_combined,1), size(X_combined,2));
fprintf('  - MFCC features:      %d columns (1-78)\n', size(X,2));
fprintf('  - Spectral features:  %d columns (79-86)\n', size(myFeatures,2));

%% Step 4: Add feature names for reference
fprintf('\n========================================\n');
fprintf('STEP 4: Creating feature labels\n');
fprintf('========================================\n');

% Create names for all 86 features
featureNames = cell(1, 86);

% MFCC feature names (1-78)
for i = 1:78
    featureNames{i} = sprintf('MFCC_%d', i);
end

% Spectral feature names (79-86) - based on Farhan's extraction
featureNames{79} = 'Centroid_Mean';
featureNames{80} = 'Centroid_Std';
featureNames{81} = 'Flux_Mean';
featureNames{82} = 'Flux_Std';
featureNames{83} = 'ZCR_Mean';
featureNames{84} = 'ZCR_Std';
featureNames{85} = 'Pitch_Mean';
featureNames{86} = 'Pitch_Std';

fprintf('Feature names created for 86 features\n');

%% Step 5: Check class distribution
fprintf('\n========================================\n');
fprintf('STEP 5: Class distribution\n');
fprintf('========================================\n');

numBonafide = sum(Y == 0);
numSpoof = sum(Y == 1);

fprintf('Bonafide (genuine): %d files (%.1f%%)\n', numBonafide, 100*numBonafide/length(Y));
fprintf('Spoof (fake):      %d files (%.1f%%)\n', numSpoof, 100*numSpoof/length(Y));

if abs(numBonafide - numSpoof) / length(Y) < 0.1
    fprintf('✓ Dataset is well-balanced\n');
else
    fprintf('⚠ Dataset is imbalanced - consider class weights in classification\n');
end

%% Step 6: Quick validation - check for any NaN or Inf values
fprintf('\n========================================\n');
fprintf('STEP 6: Data validation\n');
fprintf('========================================\n');

if any(isnan(X_combined(:)))
    fprintf('⚠ Warning: Found NaN values in combined features\n');
else
    fprintf('✓ No NaN values detected\n');
end

if any(isinf(X_combined(:)))
    fprintf('⚠ Warning: Found Inf values in combined features\n');
else
    fprintf('✓ No Inf values detected\n');
end

% Check feature ranges
fprintf('\nFeature value ranges:\n');
fprintf('  MFCC features:      [%.2f, %.2f]\n', min(X(:)), max(X(:)));
fprintf('  Spectral features:  [%.2f, %.2f]\n', min(myFeatures(:)), max(myFeatures(:)));

%% Step 7: Save the combined feature matrix
fprintf('\n========================================\n');
fprintf('STEP 7: Saving combined features\n');
fprintf('========================================\n');

save('combined_features.mat', 'X_combined', 'Y', 'featureNames', 'X', 'myFeatures');

fprintf('Saved to: combined_features.mat\n');
fprintf('File contents:\n');
fprintf('  - X_combined:  %d × %d (features)\n', size(X_combined,1), size(X_combined,2));
fprintf('  - Y:           %d × 1 (labels)\n', length(Y), 1);
fprintf('  - featureNames: 1 × %d cell array\n', length(featureNames));
fprintf('  - X:           Original MFCC features (backup)\n');
fprintf('  - myFeatures:  Original spectral features (backup)\n');

%% Step 8: Quick summary display
fprintf('\n========================================\n');
fprintf('SUMMARY - FEATURE COMBINATION COMPLETE\n');
fprintf('========================================\n');
fprintf('Total files:      %d\n', size(X_combined,1));
fprintf('Total features:   %d\n', size(X_combined,2));
fprintf('  - From Khizer:  %d (MFCCs)\n', size(X,2));
fprintf('  - From Farhan:  %d (Spectral)\n', size(myFeatures,2));
fprintf('\nOutput file:      combined_features.mat\n');
fprintf('Ready for classification!\n');
fprintf('========================================\n');

%% Optional: Display first few rows
disp(' ');
disp('First 5 files - First 10 features (preview):');
disp('(File, MFCC1-5, Centroid_Mean, Flux_Mean, ZCR_Mean, Pitch_Mean)');
preview = [X_combined(1:5, 1:5), X_combined(1:5, 79), X_combined(1:5, 81), ...
           X_combined(1:5, 83), X_combined(1:5, 85)];
disp(preview);