%% CLASSIFICATION FOR EXTREMELY IMBALANCED DATASET (8.8:1)
clear; clc; close all;

%% Load data
load('combined_features.mat');  % X_combined, Y

n0 = sum(Y==0);
n1 = sum(Y==1);
fprintf('========================================\n');
fprintf('EXTREME IMBALANCE DETECTED!\n');
fprintf('========================================\n');
fprintf('Bonafide (0): %d files (%.1f%%)\n', n0, 100*n0/length(Y));
fprintf('Spoof (1):    %d files (%.1f%%)\n', n1, 100*n1/length(Y));
fprintf('Ratio: %.1f:1 (Spoof:Bonafide)\n', n1/n0);
fprintf('\n⚠ WARNING: A dummy classifier predicting ALL as Spoof\n');
fprintf('   would achieve %.1f%% accuracy!\n', 100*n1/length(Y));
fprintf('→ Do NOT use Accuracy as your primary metric!\n');
fprintf('→ Use F1-Score, Precision, Recall, and AUC instead.\n');

%% SOLUTION 1: Balanced Training Set (Undersample majority)
fprintf('\n========================================\n');
fprintf('SOLUTION 1: Balanced Undersampling\n');
fprintf('========================================\n');

rng(42);
bonafideIdx = find(Y==0);
spoofIdx = find(Y==1);

% Take ALL bonafide samples and EQUAL number of spoof samples
n_balanced = length(bonafideIdx);
spoof_balanced = spoofIdx(randperm(length(spoofIdx), n_balanced));

balancedIdx = [bonafideIdx; spoof_balanced];
fprintf('Balanced dataset: %d bonafide + %d spoof = %d total\n', ...
    n_balanced, n_balanced, length(balancedIdx));

% Split balanced set into train/test (80/20)
p = randperm(length(balancedIdx));
trainBalanced = balancedIdx(p(1:round(0.8*length(balancedIdx))));
testBalanced = balancedIdx(p(round(0.8*length(balancedIdx))+1:end));

% Standardize
[X_bal_train, mu, sigma] = zscore(X_combined(trainBalanced, :));
X_bal_test = (X_combined(testBalanced, :) - mu) ./ sigma;

% Train SVM on balanced data
svm_balanced = fitcsvm(X_bal_train, Y(trainBalanced), ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'ClassNames', [0, 1]);

pred_balanced = predict(svm_balanced, X_bal_test);

%% SOLUTION 2: Cost-Sensitive Learning (Higher penalty for bonafide errors)
fprintf('\n========================================\n');
fprintf('SOLUTION 2: Cost-Sensitive Learning\n');
fprintf('========================================\n');

% Cost matrix: [Cost of predicting 0 when true is 0, Cost of predicting 0 when true is 1;
%               Cost of predicting 1 when true is 0, Cost of predicting 1 when true is 1]
% We penalize misclassifying bonafide (false spoof) MORE than misclassifying spoof

cost_matrix = [0, 1.0;   % Predicting 0 (bonafide): cost 0 if correct, 1.0 if wrong (spoof→bonafide)
               8.8, 0];  % Predicting 1 (spoof):   cost 8.8 if wrong (bonafide→spoof), 0 if correct
                         % 8.8 = imbalance ratio (penalize false spoof heavily)

fprintf('Cost matrix (rows=true, cols=predicted):\n');
fprintf('               Predicted\n');
fprintf('               Bonafide  Spoof\n');
fprintf('Actual Bonafide  %3.1f      %3.1f\n', cost_matrix(1,1), cost_matrix(1,2));
fprintf('       Spoof      %3.1f      %3.1f\n', cost_matrix(2,1), cost_matrix(2,2));
fprintf('\n→ Misclassifying BONAFIDE as SPOOF costs %.1fx more\n', cost_matrix(1,2)/cost_matrix(2,1));

% Use standard train/test split (preserve imbalance)
bonafideIdx = find(Y==0);
spoofIdx = find(Y==1);
bonafideIdx = bonafideIdx(randperm(length(bonafideIdx)));
spoofIdx = spoofIdx(randperm(length(spoofIdx)));

split0 = round(0.8 * length(bonafideIdx));
split1 = round(0.8 * length(spoofIdx));

trainIdx = [bonafideIdx(1:split0); spoofIdx(1:split1)];
testIdx = [bonafideIdx(split0+1:end); spoofIdx(split1+1:end)];

[X_train, mu, sigma] = zscore(X_combined(trainIdx, :));
X_test = (X_combined(testIdx, :) - mu) ./ sigma;

% Train with cost matrix
svm_cost = fitcsvm(X_train, Y(trainIdx), ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'ClassNames', [0, 1], ...
    'Cost', cost_matrix);

pred_cost = predict(svm_cost, X_test);

%% SOLUTION 3: SMOTE (Synthetic oversampling of minority class)
fprintf('\n========================================\n');
fprintf('SOLUTION 3: SMOTE Oversampling\n');
fprintf('========================================\n');

% Simple synthetic oversampling (since SMOTE may not be available)
bonafide_train = X_train(Y(trainIdx)==0, :);
spoof_train = X_train(Y(trainIdx)==1, :);

% Desired ratio: 50/50
target_bonafide = size(spoof_train, 1);
oversample_factor = ceil(target_bonafide / size(bonafide_train, 1));

% Generate synthetic bonafide samples (add noise)
synthetic_bonafide = [];
for i = 1:oversample_factor - 1
    noise = randn(size(bonafide_train)) * 0.1;  % 10% noise
    synthetic_bonafide = [synthetic_bonafide; bonafide_train + noise];
end

% Combine
X_smote = [bonafide_train; synthetic_bonafide; spoof_train];
Y_smote = [zeros(size(bonafide_train,1)+size(synthetic_bonafide,1), 1); 
           ones(size(spoof_train,1), 1)];

fprintf('Original training: %d bonafide, %d spoof\n', size(bonafide_train,1), size(spoof_train,1));
fprintf('After SMOTE-like:  %d bonafide, %d spoof\n', size(X_smote(Y_smote==0,:),1), size(X_smote(Y_smote==1,:),1));

svm_smote = fitcsvm(X_smote, Y_smote, 'KernelFunction', 'rbf', 'ClassNames', [0,1]);
pred_smote = predict(svm_smote, X_test);

%% Compare all methods
fprintf('\n========================================\n');
fprintf('COMPARISON RESULTS\n');
fprintf('========================================\n');

calc_metrics = @(pred, true) compute_metrics(pred, true);

metrics_balanced = calc_metrics(pred_balanced, Y(testBalanced));
metrics_cost = calc_metrics(pred_cost, Y(testIdx));
metrics_smote = calc_metrics(pred_smote, Y(testIdx));

fprintf('\n%-20s | %8s | %8s | %8s | %8s | %8s\n', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Set');
fprintf('%-20s-+-%8s-+-%8s-+-%8s-+-%8s-+-%8s\n', repmat('-',20,1), repmat('-',8,1), repmat('-',8,1), repmat('-',8,1), repmat('-',8,1), repmat('-',8,1));
fprintf('%-20s | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%% | %7s\n', 'Balanced', ...
    metrics_balanced.acc*100, metrics_balanced.prec*100, metrics_balanced.rec*100, metrics_balanced.f1*100, 'Balanced');
fprintf('%-20s | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%% | %7s\n', 'Cost-Sensitive', ...
    metrics_cost.acc*100, metrics_cost.prec*100, metrics_cost.rec*100, metrics_cost.f1*100, 'Original');
fprintf('%-20s | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%% | %7s\n', 'SMOTE/Oversample', ...
    metrics_smote.acc*100, metrics_smote.prec*100, metrics_smote.rec*100, metrics_smote.f1*100, 'Original');

%% Best model selection
all_f1 = [metrics_balanced.f1, metrics_cost.f1, metrics_smote.f1];
[best_f1, best_idx] = max(all_f1);
methods = {'Balanced Undersampling', 'Cost-Sensitive', 'SMOTE'};

fprintf('\n✅ BEST METHOD: %s (F1-Score = %.2f%%)\n', methods{best_idx}, best_f1*100);

%% Visualize confusion matrices
figure('Position', [100, 100, 1400, 400]);

% Balanced method confusion
subplot(1,3,1);
cm_bal = confusionmat(Y(testBalanced), pred_balanced);
confusionchart(cm_bal, {'Bonafide', 'Spoof'});
title(sprintf('Balanced Undersampling\nF1=%.1f%%', metrics_balanced.f1*100));

% Cost-sensitive confusion
subplot(1,3,2);
cm_cost = confusionmat(Y(testIdx), pred_cost);
confusionchart(cm_cost, {'Bonafide', 'Spoof'});
title(sprintf('Cost-Sensitive (Spend=8.8x)\nF1=%.1f%%', metrics_cost.f1*100));

% SMOTE confusion
subplot(1,3,3);
cm_smote = confusionmat(Y(testIdx), pred_smote);
confusionchart(cm_smote, {'Bonafide', 'Spoof'});
title(sprintf('SMOTE Oversampling\nF1=%.1f%%', metrics_smote.f1*100));

sgtitle('Imbalance Handling Methods Comparison (8.8:1 Spoof:Bonafide)');

%% Final recommendations for project report
fprintf('\n========================================\n');
fprintf('RECOMMENDATIONS FOR EAR-WITNESS PROJECT\n');
fprintf('========================================\n');

fprintf('\n1. REPORTING METRICS:\n');
fprintf('   ❌ DO NOT report Accuracy alone (will be ~90%% even for useless model)\n');
fprintf('   ✅ Report F1-Score, Precision (Spoof), Recall (Spoof)\n');
fprintf('   ✅ Show Confusion Matrix\n');
fprintf('   ✅ Report AUC-ROC\n');

fprintf('\n2. METHOD JUSTIFICATION:\n');
fprintf('   Use %s because:\n', methods{best_idx});
switch best_idx
    case 1
        fprintf('   - Balances the dataset to 50/50\n');
        fprintf('   - Removes bias toward spoof class\n');
        fprintf('   - Sacrifices some data but yields reliable metrics\n');
    case 2
        fprintf('   - Penalizes false spoof detection 8.8x more\n');
        fprintf('   - Preserves all training data\n');
        fprintf('   - Better for real-world deployment\n');
    case 3
        fprintf('   - Creates synthetic bonafide samples\n');
        fprintf('   - Preserves all original data\n');
        fprintf('   - Helps model learn minority class patterns\n');
end

fprintf('\n3. FOR YOUR PROJECT CONCLUSION:\n');
fprintf('   "Due to severe dataset imbalance (89%% spoof, 11%% bonafide),\n');
fprintf('    traditional accuracy is misleading. Our model achieves\n');
fprintf('    %.1f%% F1-Score using %s, demonstrating effective\n', best_f1*100, methods{best_idx});
fprintf('    spoof detection while controlling false alarms."\n');

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
end