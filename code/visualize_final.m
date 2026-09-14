%% COMPLETE VISUALIZATION FOR PROJECT REPORT
clear; clc; close all;

%% Load the trained model and data
load('final_svm_model.mat');     % svm_final, mu, sigma, metrics
load('combined_features.mat');   % X_combined, Y
load('test_indices.mat');        % testIdx, trainIdx (if saved)

% If testIdx not saved, recreate it
if ~exist('testIdx', 'var')
    rng(42);
    bonafideIdx = find(Y==0);
    spoofIdx = find(Y==1);
    bonafideIdx = bonafideIdx(randperm(length(bonafideIdx)));
    spoofIdx = spoofIdx(randperm(length(spoofIdx)));
    split0 = round(0.8 * length(bonafideIdx));
    split1 = round(0.8 * length(spoofIdx));
    testIdx = [bonafideIdx(split0+1:end); spoofIdx(split1+1:end)];
end

% Prepare test data
X_test = (X_combined(testIdx, :) - mu) ./ sigma;
Y_test = Y(testIdx);

% Get predictions and scores
[pred, scores] = predict(svm_final, X_test);
if size(scores, 2) == 2
    spoof_scores = scores(:, 2);
else
    spoof_scores = scores;
end

% Calculate confusion matrix
tp = sum(pred == 1 & Y_test == 1);
tn = sum(pred == 0 & Y_test == 0);
fp = sum(pred == 1 & Y_test == 0);
fn = sum(pred == 0 & Y_test == 1);

cm = [tn, fp; fn, tp];

%% FIGURE 1: Confusion Matrix (Main Result)
figure('Position', [100, 100, 550, 500]);
confusionchart(cm, {'Bonafide (Real)', 'Spoof (Fake)'}, ...
    'Title', 'SMOTE + Linear SVM - Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
xlabel('Predicted Label');
ylabel('True Label');

% Add metrics annotation
annotation('textbox', [0.15, 0.02, 0.7, 0.08], ...
    'String', sprintf('Recall: %.1f%% | False Alarm: %.1f%% | F1-Score: %.1f%%', ...
    recall*100, false_alarm*100, f1*100), ...
    'HorizontalAlignment', 'center', 'FontSize', 11, ...
    'EdgeColor', 'none', 'BackgroundColor', [0.95, 0.95, 0.95]);

saveas(gcf, 'Figure1_ConfusionMatrix.png');
fprintf('✓ Saved: Figure1_ConfusionMatrix.png\n');

%% FIGURE 2: ROC Curve
figure('Position', [100, 100, 550, 500]);
[X_roc, Y_roc, ~, auc] = perfcurve(Y_test, spoof_scores, 1);

plot(X_roc, Y_roc, 'b-', 'LineWidth', 2.5);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
xlabel('False Positive Rate (1 - Specificity)');
ylabel('True Positive Rate (Recall)');
title(sprintf('ROC Curve - AUC = %.3f', auc));
legend('SMOTE + Linear SVM', 'Random Classifier', 'Location', 'southeast');
grid on;
set(gca, 'FontSize', 11);

saveas(gcf, 'Figure2_ROCCurve.png');
fprintf('✓ Saved: Figure2_ROCCurve.png\n');

%% FIGURE 3: Performance Metrics Bar Chart
figure('Position', [100, 100, 700, 500]);

metrics_names = {'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score'};
metrics_values = [accuracy, precision, recall, specificity, f1];

% Color coding
colors = [0.2, 0.6, 0.8;   % Accuracy - blue
          0.3, 0.7, 0.4;   % Precision - green
          0.9, 0.4, 0.3;   % Recall - red
          0.6, 0.4, 0.7;   % Specificity - purple
          0.9, 0.7, 0.2];  % F1-Score - orange

for i = 1:5
    bar(i, metrics_values(i)*100, 'FaceColor', colors(i,:), 'EdgeColor', 'k', 'LineWidth', 1);
    hold on;
end

set(gca, 'XTickLabel', metrics_names);
ylabel('Percentage (%)');
title('Model Performance Metrics');
ylim([0 105]);
grid on;

% Add value labels on top of bars
for i = 1:5
    text(i, metrics_values(i)*100 + 2, sprintf('%.1f%%', metrics_values(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, 'Figure3_PerformanceMetrics.png');
fprintf('✓ Saved: Figure3_PerformanceMetrics.png\n');

%% FIGURE 4: Precision-Recall Curve
figure('Position', [100, 100, 550, 500]);
[X_pr, Y_pr, ~, pr_auc] = perfcurve(Y_test, spoof_scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');

plot(X_pr, Y_pr, 'g-', 'LineWidth', 2.5);
xlabel('Recall');
ylabel('Precision');
title(sprintf('Precision-Recall Curve - AUC = %.3f', pr_auc));
grid on;
set(gca, 'FontSize', 11);

saveas(gcf, 'Figure4_PrecisionRecall.png');
fprintf('✓ Saved: Figure4_PrecisionRecall.png\n');

%% FIGURE 5: Score Distribution (Confidence Analysis)
figure('Position', [100, 100, 700, 500]);

% Separate scores by true class
bonafide_scores = spoof_scores(Y_test == 0);
spoof_scores_class = spoof_scores(Y_test == 1);

histogram(bonafide_scores, 'FaceColor', 'b', 'EdgeColor', 'k', 'BinWidth', 0.05, 'DisplayName', 'Bonafide (Real)');
hold on;
histogram(spoof_scores_class, 'FaceColor', 'r', 'EdgeColor', 'k', 'BinWidth', 0.05, 'DisplayName', 'Spoof (Fake)');
xlabel('SVM Decision Score (higher = more likely Spoof)');
ylabel('Number of Files');
title('Score Distribution: Bonafide vs Spoof');
legend('Location', 'northwest');
grid on;

% Add threshold line at 0.5
xline(0.5, 'k--', 'LineWidth', 1.5, 'Label', 'Decision Threshold (0.5)');

% Add annotation showing overlap region
xlim([-0.5, 1.5]);
annotation('textbox', [0.55, 0.7, 0.3, 0.1], ...
    'String', sprintf('Overlap Region\n(False Alarms: %.1f%%)', false_alarm*100), ...
    'BackgroundColor', [1, 1, 0.8], 'EdgeColor', 'k');

saveas(gcf, 'Figure5_ScoreDistribution.png');
fprintf('✓ Saved: Figure5_ScoreDistribution.png\n');

%% FIGURE 6: Summary Dashboard (All-in-One)
figure('Position', [100, 100, 1200, 700]);

% Subplot 1: Confusion Matrix (small version)
subplot(2,3,1);
imagesc(cm);
colormap(flipud(gray));
colorbar;
set(gca, 'XTick', [1,2], 'XTickLabel', {'Bonafide', 'Spoof'});
set(gca, 'YTick', [1,2], 'YTickLabel', {'Bonafide', 'Spoof'});
xlabel('Predicted'); ylabel('True');
title('Confusion Matrix');
for i = 1:2
    for j = 1:2
        text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', ...
            'Color', 'red', 'FontWeight', 'bold', 'FontSize', 14);
    end
end

% Subplot 2: Metrics
subplot(2,3,2);
bar(metrics_values * 100);
set(gca, 'XTickLabel', metrics_names);
ylabel('%');
title('Performance Metrics');
ylim([0 105]);
grid on;
for i = 1:5
    text(i, metrics_values(i)*100 + 2, sprintf('%.1f', metrics_values(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% Subplot 3: ROC Curve (small)
subplot(2,3,3);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--');
xlabel('FPR'); ylabel('TPR');
title(sprintf('ROC (AUC = %.3f)', auc));
grid on;

% Subplot 4: Precision-Recall (small)
subplot(2,3,4);
plot(X_pr, Y_pr, 'g-', 'LineWidth', 2);
xlabel('Recall'); ylabel('Precision');
title(sprintf('PR Curve (AUC = %.3f)', pr_auc));
grid on;

% Subplot 5: Dataset Imbalance (Before SMOTE)
subplot(2,3,5);
before_counts = [sum(Y(trainIdx)==0), sum(Y(trainIdx)==1)];
bar(before_counts, 'FaceColor', [0.8, 0.3, 0.3]);
set(gca, 'XTickLabel', {'Bonafide', 'Spoof'});
ylabel('Number of Files');
title('Original Training Set (Imbalanced)');
grid on;

% Subplot 6: Balanced Dataset (After SMOTE)
subplot(2,3,6);
after_counts = [sum(Y_balanced==0), sum(Y_balanced==1)];
bar(after_counts, 'FaceColor', [0.3, 0.7, 0.3]);
set(gca, 'XTickLabel', {'Bonafide', 'Spoof'});
ylabel('Number of Files');
title('After SMOTE (Balanced)');
grid on;

sgtitle('EAR-WITNESS PROJECT: Complete Results Dashboard', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, 'Figure6_CompleteDashboard.png');
fprintf('✓ Saved: Figure6_CompleteDashboard.png\n');

%% FIGURE 7: Feature Importance (Top 20 features)
% Get feature names if available
if exist('featureNames', 'var')
    feature_names = featureNames;
else
    % Create generic names
    feature_names = cell(1, size(svm_final.Beta, 1));
    for i = 1:78
        feature_names{i} = sprintf('MFCC_%d', i);
    end
    feature_names{79} = 'Centroid_Mean';
    feature_names{80} = 'Centroid_Std';
    feature_names{81} = 'Flux_Mean';
    feature_names{82} = 'Flux_Std';
    feature_names{83} = 'ZCR_Mean';
    feature_names{84} = 'ZCR_Std';
    feature_names{85} = 'Pitch_Mean';
    feature_names{86} = 'Pitch_Std';
end

% Get absolute weights
[weights_sorted, idx_sorted] = sort(abs(svm_final.Beta), 'descend');

% Plot top 20
figure('Position', [100, 100, 800, 500]);
bar(weights_sorted(1:20));
xlabel('Feature Index');
ylabel('Absolute SVM Weight');
title('Top 20 Most Important Features');
grid on;

% Add feature names if space permits
set(gca, 'XTick', 1:5:20);
set(gca, 'XTickLabel', 1:5:20);

saveas(gcf, 'Figure7_FeatureImportance.png');
fprintf('✓ Saved: Figure7_FeatureImportance.png\n');

%% Generate a summary text file
fid = fopen('Project_Results_Summary.txt', 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'EAR-WITNESS PROJECT - FINAL RESULTS\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'DATASET STATISTICS:\n');
fprintf(fid, '  Total files:      %d\n', length(Y));
fprintf(fid, '  Bonafide (real):  %d (%.1f%%)\n', sum(Y==0), 100*sum(Y==0)/length(Y));
fprintf(fid, '  Spoof (fake):     %d (%.1f%%)\n', sum(Y==1), 100*sum(Y==1)/length(Y));
fprintf(fid, '  Imbalance ratio:  %.1f:1 (Spoof:Bonafide)\n\n', sum(Y==1)/sum(Y==0));

fprintf(fid, 'METHOD:\n');
fprintf(fid, '  Imbalance handling: SMOTE (Synthetic Minority Oversampling)\n');
fprintf(fid, '  Classifier:         Linear SVM\n');
fprintf(fid, '  Features:           86 (78 MFCC + 8 Spectral)\n');
fprintf(fid, '  Train/Test split:   80/20 (stratified)\n\n');

fprintf(fid, 'RESULTS:\n');
fprintf(fid, '  Accuracy:           %.2f%%\n', accuracy*100);
fprintf(fid, '  Precision:          %.2f%%\n', precision*100);
fprintf(fid, '  Recall (Spoof):     %.2f%%\n', recall*100);
fprintf(fid, '  Specificity:        %.2f%%\n', specificity*100);
fprintf(fid, '  False Alarm Rate:   %.2f%%\n', false_alarm*100);
fprintf(fid, '  F1-Score:           %.2f%%\n', f1*100);
fprintf(fid, '  AUC-ROC:            %.3f\n', auc);
fprintf(fid, '  PR-AUC:             %.3f\n\n', pr_auc);

fprintf(fid, 'CONFUSION MATRIX:\n');
fprintf(fid, '                 Predicted\n');
fprintf(fid, '                 Bonafide  Spoof\n');
fprintf(fid, 'Actual Bonafide    %4d     %4d\n', tn, fp);
fprintf(fid, '       Spoof        %4d     %4d\n\n', fn, tp);

fprintf(fid, 'INTERPRETATION:\n');
fprintf(fid, '  - Spoof detection rate: %.1f%% (catches most fake audio)\n', recall*100);
fprintf(fid, '  - False alarm rate: %.1f%% (%.0f%% reduction from baseline)\n', false_alarm*100, 100 - false_alarm*100/0.55);
fprintf(fid, '  - Model is suitable for research prototype deployment\n');
fprintf(fid, '  - For production: consider confidence threshold or human review\n');

fclose(fid);
fprintf('\n✓ Saved: Project_Results_Summary.txt\n');

%% Display final summary
fprintf('\n========================================\n');
fprintf('ALL FIGURES GENERATED\n');
fprintf('========================================\n');
fprintf('1. Figure1_ConfusionMatrix.png\n');
fprintf('2. Figure2_ROCCurve.png\n');
fprintf('3. Figure3_PerformanceMetrics.png\n');
fprintf('4. Figure4_PrecisionRecall.png\n');
fprintf('5. Figure5_ScoreDistribution.png\n');
fprintf('6. Figure6_CompleteDashboard.png\n');
fprintf('7. Figure7_FeatureImportance.png\n');
fprintf('8. Project_Results_Summary.txt\n');
fprintf('\nAll files saved in current folder.\n');