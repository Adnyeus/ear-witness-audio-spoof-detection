%% MAIN TRAINING SCRIPT - Cost-Sensitive SVM
clear; clc; close all;

%% Load combined features
load('combined_features.mat');  % X_combined, Y

%% Dataset info
n0 = sum(Y==0);
n1 = sum(Y==1);
imbalance_ratio = n1 / n0;
fprintf('Dataset: %d bonafide, %d spoof (%.1f:1 ratio)\n', n0, n1, imbalance_ratio);

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

%% Standardize
[X_train, mu, sigma] = zscore(X_combined(trainIdx, :));
X_test = (X_combined(testIdx, :) - mu) ./ sigma;

%% Train with cost matrix (using imbalance ratio)
cost_matrix = [0, imbalance_ratio; 1, 0];
fprintf('Cost matrix: False Spoof penalty = %.1fx\n', imbalance_ratio);

svm_model = fitcsvm(X_train, Y(trainIdx), ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'ClassNames', [0, 1], ...
    'Cost', cost_matrix);

%% Evaluate
pred = predict(svm_model, X_test);
metrics = compute_metrics(pred, Y(testIdx));

fprintf('\nRESULTS:\n');
fprintf('  Accuracy:  %.2f%%\n', metrics.acc * 100);
fprintf('  Precision: %.2f%%\n', metrics.prec * 100);
fprintf('  Recall:    %.2f%%\n', metrics.rec * 100);
fprintf('  F1-Score:  %.2f%%\n', metrics.f1 * 100);

%% Save model and preprocessing params
save('trained_svm_model.mat', 'svm_model', 'mu', 'sigma', 'metrics');
save('test_indices.mat', 'testIdx', 'trainIdx');
fprintf('\n✓ Model saved to trained_svm_model.mat\n');

%% Helper function
function metrics = compute_metrics(pred, true)
    tp = sum(pred == 1 & true == 1);
    tn = sum(pred == 0 & true == 0);
    fp = sum(pred == 1 & true == 0);
    fn = sum(pred == 0 & true == 1);
    
    metrics.acc = (tp + tn) / length(true);
    metrics.prec = tp / (tp + fp + eps);
    metrics.rec = tp / (tp + fn + eps);
    metrics.f1 = 2 * metrics.prec * metrics.rec / (metrics.prec + metrics.rec + eps);
    metrics.tp = tp;
    metrics.tn = tn;
    metrics.fp = fp;
    metrics.fn = fn;
end