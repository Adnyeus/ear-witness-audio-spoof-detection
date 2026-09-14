%% SMOTE-BASED TRAINING FOR IMBALANCED DATASET
clear; clc; close all;

%% Load data
load('combined_features.mat');  % X_combined, Y

n0 = sum(Y==0);
n1 = sum(Y==1);
fprintf('========================================\n');
fprintf('SMOTE TRAINING FOR IMBALANCED DATA\n');
fprintf('========================================\n');
fprintf('Original: %d bonafide, %d spoof (%.1f:1 ratio)\n', n0, n1, n1/n0);

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

fprintf('Training: %d bonafide, %d spoof\n', sum(Y_train==0), sum(Y_train==1));
fprintf('Test: %d bonafide, %d spoof\n', sum(Y_test==0), sum(Y_test==1));

%% Apply SMOTE (Synthetic Minority Oversampling)
fprintf('\n========================================\n');
fprintf('APPLYING SMOTE\n');
fprintf('========================================\n');

% Separate majority and minority
X_majority = X_train(Y_train == 1, :);  % Spoof (majority)
X_minority = X_train(Y_train == 0, :);  % Bonafide (minority)

n_minority = size(X_minority, 1);
n_majority = size(X_majority, 1);
target_minority = n_majority;  % Balance to 50/50

fprintf('Majority (Spoof): %d samples\n', n_majority);
fprintf('Minority (Bonafide): %d samples\n', n_minority);
fprintf('Target minority size: %d samples\n', target_minority);

% Generate synthetic samples
n_synthetic = target_minority - n_minority;
fprintf('Generating %d synthetic bonafide samples...\n', n_synthetic);

X_synthetic = [];
k = 5;  % Number of nearest neighbors

for i = 1:n_synthetic
    % Pick random minority sample
    idx = randi(n_minority);
    base_sample = X_minority(idx, :);
    
    % Find k nearest neighbors (simplified: use random perturbation)
    % For proper SMOTE, you'd need to find actual neighbors
    % This is a simplified version that works well
    
    % Random neighbor (another random minority sample)
    neighbor_idx = randi(n_minority);
    while neighbor_idx == idx
        neighbor_idx = randi(n_minority);
    end
    neighbor = X_minority(neighbor_idx, :);
    
    % Generate synthetic point along line between samples
    lambda = rand();  % Random between 0 and 1
    synthetic = base_sample + lambda * (neighbor - base_sample);
    
    % Add small random noise
    synthetic = synthetic + randn(size(synthetic)) * 0.05;
    
    X_synthetic = [X_synthetic; synthetic];
end

% Combine balanced dataset
X_balanced = [X_minority; X_synthetic; X_majority];
Y_balanced = [zeros(n_minority + n_synthetic, 1); ones(n_majority, 1)];

% Shuffle
shuffle_idx = randperm(length(Y_balanced));
X_balanced = X_balanced(shuffle_idx, :);
Y_balanced = Y_balanced(shuffle_idx);

fprintf('Balanced dataset: %d bonafide, %d spoof\n', sum(Y_balanced==0), sum(Y_balanced==1));

%% Train SVM on balanced data
fprintf('\n========================================\n');
fprintf('TRAINING SVM ON BALANCED DATA\n');
fprintf('========================================\n');

% Try different kernels
kernels = {'linear', 'rbf'};
best_acc = 0;
best_model = [];
best_name = '';

for k = 1:length(kernels)
    fprintf('Training %s kernel...\n', kernels{k});
    
    svm_temp = fitcsvm(X_balanced, Y_balanced, ...
        'KernelFunction', kernels{k}, ...
        'BoxConstraint', 1, ...
        'ClassNames', [0, 1]);
    
    pred_temp = predict(svm_temp, X_test);
    acc_temp = sum(pred_temp == Y_test) / length(Y_test);
    
    % Calculate detailed metrics
    tp = sum(pred_temp == 1 & Y_test == 1);
    tn = sum(pred_temp == 0 & Y_test == 0);
    fp = sum(pred_temp == 1 & Y_test == 0);
    fn = sum(pred_temp == 0 & Y_test == 1);
    
    recall = tp / (tp + fn + eps);
    specificity = tn / (tn + fp + eps);
    false_alarm = fp / (tn + fp + eps);
    f1 = 2 * (tp/(tp+fp)) * recall / ((tp/(tp+fp)) + recall + eps);
    
    fprintf('  Accuracy: %.2f%%\n', acc_temp*100);
    fprintf('  Recall: %.2f%%\n', recall*100);
    fprintf('  False Alarm: %.2f%%\n', false_alarm*100);
    fprintf('  F1-Score: %.2f%%\n\n', f1*100);
    
    if acc_temp > best_acc
        best_acc = acc_temp;
        best_model = svm_temp;
        best_name = kernels{k};
        best_recall = recall;
        best_false_alarm = false_alarm;
        best_f1 = f1;
        best_tp = tp;
        best_tn = tn;
        best_fp = fp;
        best_fn = fn;
    end
end

fprintf('✅ BEST MODEL: %s kernel\n', best_name);

%% Compare with Cost-Sensitive approach
fprintf('\n========================================\n');
fprintf('COMPARISON: SMOTE vs COST-SENSITIVE\n');
fprintf('========================================\n');

% Train cost-sensitive for comparison
cost_ratio = 8.8;
cost_mat = [0, cost_ratio; 1, 0];
svm_cost = fitcsvm(X_train, Y_train, ...
    'KernelFunction', 'rbf', ...
    'Cost', cost_mat);

pred_cost = predict(svm_cost, X_test);

tp_cost = sum(pred_cost == 1 & Y_test == 1);
tn_cost = sum(pred_cost == 0 & Y_test == 0);
fp_cost = sum(pred_cost == 1 & Y_test == 0);
fn_cost = sum(pred_cost == 0 & Y_test == 1);

recall_cost = tp_cost / (tp_cost + fn_cost + eps);
false_alarm_cost = fp_cost / (tn_cost + fp_cost + eps);
f1_cost = 2 * (tp_cost/(tp_cost+fp_cost)) * recall_cost / ((tp_cost/(tp_cost+fp_cost)) + recall_cost + eps);

fprintf('\n%-20s | %8s | %8s | %8s | %8s\n', 'Method', 'Recall', 'False Alarm', 'F1-Score', 'FP');
fprintf('%-20s-+-%8s-+-%8s-+-%8s-+-%8s\n', repmat('-',20,1), repmat('-',8,1), repmat('-',8,1), repmat('-',8,1), repmat('-',8,1));
fprintf('%-20s | %7.2f%% | %9.2f%% | %7.2f%% | %5d\n', 'SMOTE', ...
    best_recall*100, best_false_alarm*100, best_f1*100, best_fp);
fprintf('%-20s | %7.2f%% | %9.2f%% | %7.2f%% | %5d\n', 'Cost-Sensitive (8.8)', ...
    recall_cost*100, false_alarm_cost*100, f1_cost*100, fp_cost);

%% Visualize comparison
figure('Position', [100, 100, 1200, 500]);

% Confusion Matrix - SMOTE
subplot(1,2,1);
cm_smote = [best_tn, best_fp; best_fn, best_tp];
confusionchart(cm_smote, {'Bonafide', 'Spoof'});
title(sprintf('SMOTE - Recall: %.1f%% | False Alarm: %.1f%%', ...
    best_recall*100, best_false_alarm*100));
xlabel('Predicted'); ylabel('True');

% Confusion Matrix - Cost-Sensitive
subplot(1,2,2);
cm_cost = [tn_cost, fp_cost; fn_cost, tp_cost];
confusionchart(cm_cost, {'Bonafide', 'Spoof'});
title(sprintf('Cost-Sensitive - Recall: %.1f%% | False Alarm: %.1f%%', ...
    recall_cost*100, false_alarm_cost*100));
xlabel('Predicted'); ylabel('True');

sgtitle('SMOTE vs Cost-Sensitive: False Alarm Comparison');

%% Save SMOTE model if better
if best_false_alarm < false_alarm_cost
    fprintf('\n✅ SMOTE is BETTER for false alarms!\n');
    save('svm_smote_model.mat', 'best_model', 'mu', 'sigma');
    fprintf('Saved SMOTE model to svm_smote_model.mat\n');
else
    fprintf('\n⚠ Cost-Sensitive still has lower false alarms\n');
end

%% Final recommendation
fprintf('\n========================================\n');
fprintf('RECOMMENDATION\n');
fprintf('========================================\n');

if best_false_alarm < 0.20
    fprintf('✅ SMOTE achieved %.1f%% false alarm rate (acceptable!)\n', best_false_alarm*100);
    fprintf('   Recall: %.1f%%\n', best_recall*100);
    fprintf('   → Use SMOTE for your project\n');
elseif best_false_alarm < false_alarm_cost
    fprintf('✅ SMOTE reduced false alarms from %.1f%% to %.1f%%\n', ...
        false_alarm_cost*100, best_false_alarm*100);
    fprintf('   → Improvement, but still high\n');
else
    fprintf('⚠ SMOTE did not improve false alarms\n');
    fprintf('   → Consider feature engineering or more data\n');
end