%% FINAL PRODUCTION MODEL - SMOTE + Linear SVM
clear; clc; close all;

%% Load data
load('combined_features.mat');  % X_combined, Y

fprintf('========================================\n');
fprintf('FINAL MODEL: SMOTE + Linear SVM\n');
fprintf('========================================\n');

%% Train/Test Split (stratified)
rng(42);
bonafideIdx = find(Y==0);
spoofIdx = find(Y==1);

bonafideIdx = bonafideIdx(randperm(length(bonafideIdx)));
spoofIdx = spoofIdx(randperm(length(spoofIdx)));

split0 = round(0.8 * length(bonafideIdx));
split1 = round(0.8 * length(spoofIdx));

trainIdx = [bonafideIdx(1:split0); spoofIdx(1:split1)];
testIdx = [bonafideIdx(split0+1:end); spoofIdx(split1+1:end)];

% Standardize
[X_train, mu, sigma] = zscore(X_combined(trainIdx, :));
X_test = (X_combined(testIdx, :) - mu) ./ sigma;

Y_train = Y(trainIdx);
Y_test = Y(testIdx);

%% SMOTE - Balance the dataset
X_majority = X_train(Y_train == 1, :);  % Spoof
X_minority = X_train(Y_train == 0, :);  % Bonafide

n_minority = size(X_minority, 1);
n_majority = size(X_majority, 1);

fprintf('Original training: %d bonafide, %d spoof\n', n_minority, n_majority);

% Generate synthetic bonafide samples
n_synthetic = n_majority - n_minority;
X_synthetic = [];

for i = 1:n_synthetic
    % Pick random bonafide sample
    idx = randi(n_minority);
    base = X_minority(idx, :);
    
    % Pick random neighbor
    neighbor_idx = randi(n_minority);
    while neighbor_idx == idx
        neighbor_idx = randi(n_minority);
    end
    neighbor = X_minority(neighbor_idx, :);
    
    % Interpolate
    lambda = rand();
    synthetic = base + lambda * (neighbor - base);
    
    % Add small noise
    synthetic = synthetic + randn(size(synthetic)) * 0.05;
    
    X_synthetic = [X_synthetic; synthetic];
end

% Create balanced dataset
X_balanced = [X_minority; X_synthetic; X_majority];
Y_balanced = [zeros(n_minority + n_synthetic, 1); ones(n_majority, 1)];

% Shuffle
shuffle_idx = randperm(length(Y_balanced));
X_balanced = X_balanced(shuffle_idx, :);
Y_balanced = Y_balanced(shuffle_idx);

fprintf('After SMOTE: %d bonafide, %d spoof (balanced)\n', ...
    sum(Y_balanced==0), sum(Y_balanced==1));

%% Train Linear SVM
fprintf('\nTraining Linear SVM on balanced data...\n');
svm_final = fitcsvm(X_balanced, Y_balanced, ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', 1, ...
    'ClassNames', [0, 1]);

%% Evaluate
pred = predict(svm_final, X_test);

% Calculate metrics
tp = sum(pred == 1 & Y_test == 1);
tn = sum(pred == 0 & Y_test == 0);
fp = sum(pred == 1 & Y_test == 0);
fn = sum(pred == 0 & Y_test == 1);

accuracy = (tp + tn) / length(Y_test);
precision = tp / (tp + fp);
recall = tp / (tp + fn);
specificity = tn / (tn + fp);
false_alarm = fp / (tn + fp);
f1 = 2 * precision * recall / (precision + recall);

fprintf('\n========================================\n');
fprintf('FINAL MODEL PERFORMANCE\n');
fprintf('========================================\n');
fprintf('Accuracy:    %.2f%%\n', accuracy * 100);
fprintf('Precision:   %.2f%%\n', precision * 100);
fprintf('Recall:      %.2f%%  (Spoof detection rate)\n', recall * 100);
fprintf('Specificity: %.2f%%  (Bonafide recognition rate)\n', specificity * 100);
fprintf('False Alarm: %.2f%%  (Bonafide flagged as Spoof)\n', false_alarm * 100);
fprintf('F1-Score:    %.2f%%\n', f1 * 100);

%% Confusion Matrix
figure('Position', [100, 100, 500, 450]);
cm = [tn, fp; fn, tp];
confusionchart(cm, {'Bonafide', 'Spoof'});
title(sprintf('SMOTE + Linear SVM\nFalse Alarm: %.1f%% | Recall: %.1f%%', ...
    false_alarm*100, recall*100));
xlabel('Predicted'); ylabel('True');

%% Save final model
save('final_svm_model.mat', 'svm_final', 'mu', 'sigma', ...
    'accuracy', 'precision', 'recall', 'specificity', 'false_alarm', 'f1');
fprintf('\n✓ Final model saved to final_svm_model.mat\n');

%% Summary for report
fprintf('\n========================================\n');
fprintf('PROJECT SUMMARY\n');
fprintf('========================================\n');
fprintf('Dataset: %d total files (%d bonafide, %d spoof)\n', ...
    length(Y), sum(Y==0), sum(Y==1));
fprintf('Features: 86 (78 MFCC + 8 Spectral)\n');
fprintf('Class imbalance handled by: SMOTE (synthetic oversampling)\n');
fprintf('Classifier: Linear SVM\n');
fprintf('\nFinal Metrics:\n');
fprintf('  - Spoof Detection Rate (Recall): %.1f%%\n', recall * 100);
fprintf('  - False Alarm Rate: %.1f%%\n', false_alarm * 100);
fprintf('  - F1-Score: %.1f%%\n', f1 * 100);