%% SVM CLASSIFICATION FOR SPOOF DETECTION
% Script by: Ebad
% Description: Trains SVM classifier on combined features (86 dimensions)
%              to detect spoofed vs bonafide audio

clear; clc; close all;

%% Step 1: Load combined features
fprintf('========================================\n');
fprintf('STEP 1: Loading combined features\n');
fprintf('========================================\n');

load('combined_features.mat');  % Loads X_combined, Y, featureNames

fprintf('Loaded: %d files × %d features\n', size(X_combined,1), size(X_combined,2));
fprintf('Labels: %d bonafide, %d spoof\n', sum(Y==0), sum(Y==1));

%% Step 2: Train/Test Split (80/20 with stratification)
fprintf('\n========================================\n');
fprintf('STEP 2: Train/Test Split\n');
fprintf('========================================\n');

rng(42);  % For reproducibility

% Stratified split: maintain class proportion in both sets
n = size(X_combined, 1);
bonafideIdx = find(Y == 0);
spoofIdx = find(Y == 1);

% Shuffle within each class
bonafideIdx = bonafideIdx(randperm(length(bonafideIdx)));
spoofIdx = spoofIdx(randperm(length(spoofIdx)));

% Split each class (80/20)
splitBonafide = round(0.8 * length(bonafideIdx));
splitSpoof = round(0.8 * length(spoofIdx));

trainBonafide = bonafideIdx(1:splitBonafide);
testBonafide = bonafideIdx(splitBonafide+1:end);

trainSpoof = spoofIdx(1:splitSpoof);
testSpoof = spoofIdx(splitSpoof+1:end);

% Combine
trainIdx = [trainBonafide; trainSpoof];
testIdx = [testBonafide; testSpoof];

fprintf('Training set: %d files (%.1f%% of total)\n', length(trainIdx), 100*length(trainIdx)/n);
fprintf('  - Bonafide: %d\n', length(trainBonafide));
fprintf('  - Spoof:    %d\n', length(trainSpoof));
fprintf('Test set:     %d files (%.1f%% of total)\n', length(testIdx), 100*length(testIdx)/n);
fprintf('  - Bonafide: %d\n', length(testBonafide));
fprintf('  - Spoof:    %d\n', length(testSpoof));

%% Step 3: Standardize features (IMPORTANT for SVM)
fprintf('\n========================================\n');
fprintf('STEP 3: Feature Standardization\n');
fprintf('========================================\n');

% Standardize using training set statistics only (prevents data leakage)
[X_train, mu, sigma] = zscore(X_combined(trainIdx, :));
X_test = (X_combined(testIdx, :) - mu) ./ sigma;

fprintf('Features standardized to zero mean and unit variance\n');
fprintf('Training set mean range: [%.2f, %.2f]\n', min(mu), max(mu));
fprintf('Training set std range:  [%.2f, %.2f]\n', min(sigma), max(sigma));

%% Step 4: Train SVM with different kernels and hyperparameters
fprintf('\n========================================\n');
fprintf('STEP 4: Training SVM Classifiers\n');
fprintf('========================================\n');

% 4a: Linear SVM (baseline)
fprintf('\nTraining Linear SVM...\n');
tic;
svm_linear = fitcsvm(X_train, Y(trainIdx), ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', 1, ...
    'ClassNames', [0, 1]);
linear_time = toc;
fprintf('  Time: %.2f seconds\n', linear_time);

% 4b: RBF SVM (usually better for complex boundaries)
fprintf('\nTraining RBF SVM...\n');
tic;
svm_rbf = fitcsvm(X_train, Y(trainIdx), ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'KernelScale', 'auto', ...
    'ClassNames', [0, 1]);
rbf_time = toc;
fprintf('  Time: %.2f seconds\n', rbf_time);

% 4c: Linear SVM with cross-validation for hyperparameter tuning
fprintf('\nPerforming cross-validation for Linear SVM...\n');
cv_model = fitcsvm(X_train, Y(trainIdx), ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', 1, ...
    'ClassNames', [0, 1], ...
    'CrossVal', 'on', ...
    'KFold', 5);
cv_acc = 1 - kfoldLoss(cv_model);
fprintf('  5-fold CV accuracy: %.2f%%\n', cv_acc * 100);

%% Step 5: Predict on test set
fprintf('\n========================================\n');
fprintf('STEP 5: Testing on Unseen Data\n');
fprintf('========================================\n');

% Linear SVM predictions
pred_linear = predict(svm_linear, X_test);
acc_linear = sum(pred_linear == Y(testIdx)) / length(testIdx);

% RBF SVM predictions
pred_rbf = predict(svm_rbf, X_test);
acc_rbf = sum(pred_rbf == Y(testIdx)) / length(testIdx);

fprintf('Linear SVM Test Accuracy: %.2f%%\n', acc_linear * 100);
fprintf('RBF SVM Test Accuracy:    %.2f%%\n', acc_rbf * 100);

%% Step 6: Detailed metrics for best model
fprintf('\n========================================\n');
fprintf('STEP 6: Detailed Performance Metrics\n');
fprintf('========================================\n');

% Choose best model (linear or rbf)
if acc_linear >= acc_rbf
    best_model = svm_linear;
    best_pred = pred_linear;
    best_name = 'Linear SVM';
    best_acc = acc_linear;
else
    best_model = svm_rbf;
    best_pred = pred_rbf;
    best_name = 'RBF SVM';
    best_acc = acc_rbf;
end

fprintf('Best model: %s (Accuracy: %.2f%%)\n', best_name, best_acc * 100);

% Calculate metrics
tp = sum(best_pred == 1 & Y(testIdx) == 1);  % True Spoof
tn = sum(best_pred == 0 & Y(testIdx) == 0);  % True Bonafide
fp = sum(best_pred == 1 & Y(testIdx) == 0);  % False Spoof (false alarm)
fn = sum(best_pred == 0 & Y(testIdx) == 1);  % False Bonafide (miss)

precision = tp / (tp + fp);
recall = tp / (tp + fn);  % Also called True Positive Rate
specificity = tn / (tn + fp);  % True Negative Rate
f1 = 2 * (precision * recall) / (precision + recall);

fprintf('\nConfusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 Bonafide  Spoof\n');
fprintf('Actual Bonafide  %6d   %6d\n', tn, fp);
fprintf('       Spoof      %6d   %6d\n', fn, tp);

fprintf('\nPerformance Metrics:\n');
fprintf('  Accuracy:     %.2f%%\n', best_acc * 100);
fprintf('  Precision:    %.2f%%\n', precision * 100);
fprintf('  Recall (Spoof): %.2f%%\n', recall * 100);
fprintf('  Specificity (Bonafide): %.2f%%\n', specificity * 100);
fprintf('  F1-Score:     %.2f%%\n', f1 * 100);

%% Step 7: Visualize results
fprintf('\n========================================\n');
fprintf('STEP 7: Generating Visualizations\n');
fprintf('========================================\n');

% Figure 1: Confusion Matrix
figure('Position', [100, 100, 500, 450]);
cm = confusionmat(Y(testIdx), best_pred);
confusionchart(cm, {'Bonafide', 'Spoof'}, ...
    'Title', sprintf('%s - Accuracy: %.2f%%', best_name, best_acc*100), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
xlabel('Predicted Label');
ylabel('True Label');

% Figure 2: Feature Importance (Linear SVM only)
if strcmp(best_model.KernelParameters.Function, 'linear')
    figure('Position', [100, 100, 1000, 400]);
    
    % Get absolute weights
    [weights_sorted, idx_sorted] = sort(abs(best_model.Beta), 'descend');
    
    % Plot top 30 features
    subplot(1,2,1);
    bar(weights_sorted(1:30));
    xlabel('Feature Rank');
    ylabel('Absolute SVM Weight');
    title('Top 30 Most Important Features');
    grid on;
    
    % Feature type breakdown
    subplot(1,2,2);
    feature_types = [repmat({'MFCC'}, 78, 1); 
                     repmat({'Centroid'}, 2, 1);
                     repmat({'Flux'}, 2, 1);
                     repmat({'ZCR'}, 2, 1);
                     repmat({'Pitch'}, 2, 1)];
    
    top30_features = idx_sorted(1:30);
    top30_types = feature_types(top30_features);
    [type_names, ~, type_idx] = unique(top30_types);
    type_counts = histcounts(type_idx, 1:length(type_names)+1);
    
    bar(type_counts);
    set(gca, 'XTickLabel', type_names);
    xlabel('Feature Type');
    ylabel('Count in Top 30');
    title('Which Features Matter Most?');
    grid on;
    
    sgtitle('SVM Feature Importance Analysis');
end

% Figure 3: ROC Curve
figure('Position', [100, 100, 500, 450]);

% Get decision scores
[~, scores] = predict(best_model, X_test);

% Calculate ROC curve
if size(scores, 2) == 2
    scores_positive = scores(:, 2);  % Score for spoof class (1)
else
    scores_positive = scores;
end

[X_roc, Y_roc, ~, auc] = perfcurve(Y(testIdx), scores_positive, 1);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlabel('False Positive Rate (1 - Specificity)');
ylabel('True Positive Rate (Recall)');
title(sprintf('ROC Curve - AUC = %.3f', auc));
legend(sprintf('%s (AUC = %.3f)', best_name, auc), 'Random Classifier', 'Location', 'southeast');
grid on;

%% Step 8: Save results
fprintf('\n========================================\n');
fprintf('STEP 8: Saving Results\n');
fprintf('========================================\n');

results.best_model = best_name;
results.accuracy = best_acc;
results.precision = precision;
results.recall = recall;
results.specificity = specificity;
results.f1_score = f1;
results.auc = auc;
results.confusion_matrix = cm;
results.linear_accuracy = acc_linear;
results.rbf_accuracy = acc_rbf;
results.cv_accuracy = cv_acc;

% Save metrics
save('classification_results.mat', 'results');
fprintf('Results saved to: classification_results.mat\n');

%% Step 9: Final Summary Report
fprintf('\n========================================\n');
fprintf('FINAL SUMMARY - SPOOF DETECTION RESULTS\n');
fprintf('========================================\n');
fprintf('Dataset:        ASVspoof 2019 LA\n');
fprintf('Total files:    %d\n', n);
fprintf('Features:       %d (78 MFCC + 8 Spectral)\n', size(X_combined,2));
fprintf('Train/Test:     %d / %d (80/20 split)\n', length(trainIdx), length(testIdx));
fprintf('\nBEST CLASSIFIER: %s\n', best_name);
fprintf('Accuracy:       %.2f%%\n', best_acc * 100);
fprintf('AUC:            %.3f\n', auc);
fprintf('F1-Score:       %.2f%%\n', f1 * 100);
fprintf('\nInterpretation:\n');

if best_acc > 95
    fprintf('  ✅ Excellent - Model is highly effective at detecting spoofed audio\n');
elseif best_acc > 85
    fprintf('  ✅ Good - Model works well for practical use\n');
elseif best_acc > 75
    fprintf('  ⚠ Moderate - May need further tuning or more features\n');
else
    fprintf('  ❌ Poor - Review feature extraction or try different classifier\n');
end

if recall > 0.9 && specificity > 0.9
    fprintf('  ✅ Balanced - Model detects both bonafide and spoof equally well\n');
elseif recall > specificity
    fprintf('  ⚠ Biased toward detecting spoof (may miss some bonafide)\n');
else
    fprintf('  ⚠ Biased toward detecting bonafide (may miss some spoof)\n');
end

fprintf('\n========================================\n');
fprintf('CLASSIFICATION COMPLETE!\n');
fprintf('========================================\n');

%% Optional: Test with a single new audio file (if needed)
% Uncomment this section to test on a single file
% fprintf('\nOptional: Test on a single audio file?\n');
% response = input('Enter path to .flac file (or press Enter to skip): ', 's');
% if ~isempty(response)
%     [audio, fs] = audioread(response);
%     % ... feature extraction code would go here ...
%     % Then predict using best_model
% end