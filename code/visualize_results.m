%% VISUALIZATION SCRIPT - Generate All Figures
clear; clc; close all;

%% Load saved model and data
load('combined_features.mat');  % X_combined, Y
load('trained_svm_model.mat');  % svm_model, mu, sigma, metrics
load('test_indices.mat');       % testIdx, trainIdx

%% Recreate predictions for visualization
[X_train, ~, ~] = zscore(X_combined(trainIdx, :));
X_test = (X_combined(testIdx, :) - mu) ./ sigma;
pred = predict(svm_model, X_test);

%% Figure 1: Confusion Matrix
figure('Position', [100, 100, 500, 450]);
cm = confusionmat(Y(testIdx), pred);
confusionchart(cm, {'Bonafide', 'Spoof'});
title(sprintf('Cost-Sensitive SVM\nF1-Score = %.1f%%', metrics.f1*100));
xlabel('Predicted'); ylabel('True');

% Add text annotation
annotation('textbox', [0.15, 0.02, 0.7, 0.05], ...
    'String', sprintf('Recall: %.1f%% | Precision: %.1f%% | F1: %.1f%%', ...
    metrics.rec*100, metrics.prec*100, metrics.f1*100), ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'EdgeColor', 'none');

saveas(gcf, 'confusion_matrix.png');

%% Figure 2: ROC Curve
figure('Position', [100, 100, 500, 450]);
[~, scores] = predict(svm_model, X_test);
if size(scores, 2) == 2
    scores_pos = scores(:, 2);
else
    scores_pos = scores;
end
[X_roc, Y_roc, ~, auc] = perfcurve(Y(testIdx), scores_pos, 1);

plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlabel('False Positive Rate (1 - Specificity)');
ylabel('True Positive Rate (Recall)');
title(sprintf('ROC Curve - AUC = %.3f', auc));
legend('Cost-Sensitive SVM', 'Random Classifier', 'Location', 'southeast');
grid on;

saveas(gcf, 'roc_curve.png');

%% Figure 3: Feature Importance (Top 20)
if strcmp(svm_model.KernelParameters.Function, 'linear')
    figure('Position', [100, 100, 600, 400]);
    [w_sorted, idx_sorted] = sort(abs(svm_model.Beta), 'descend');
    bar(w_sorted(1:20));
    xlabel('Feature Rank'); ylabel('Absolute Weight');
    title('Top 20 Most Important Features');
    grid on;
    saveas(gcf, 'feature_importance.png');
else
    fprintf('RBF kernel used - feature importance not directly available\n');
end

%% Figure 4: Summary Dashboard
figure('Position', [100, 100, 800, 600]);

% Subplot 1: Confusion Matrix as heatmap
subplot(2,2,1);
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

% Subplot 2: Metrics Bar Chart
subplot(2,2,2);
metrics_names = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
metrics_values = [metrics.acc, metrics.prec, metrics.rec, metrics.f1];
bar(metrics_values * 100);
set(gca, 'XTickLabel', metrics_names);
ylabel('Percentage (%)');
title('Performance Metrics');
ylim([0 105]);
for i = 1:4
    text(i, metrics_values(i)*100 + 2, sprintf('%.1f%%', metrics_values(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
grid on;

% Subplot 3: Class Distribution
subplot(2,2,3);
class_counts = [sum(Y(testIdx)==0), sum(Y(testIdx)==1)];
bar(class_counts);
set(gca, 'XTickLabel', {'Bonafide', 'Spoof'});
ylabel('Number of Files');
title('Test Set Distribution');
grid on;

% Subplot 4: ROC Curve (small version)
subplot(2,2,4);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--');
xlabel('FPR'); ylabel('TPR');
title(sprintf('ROC (AUC = %.3f)', auc));
grid on;

sgtitle('Ear-Witness: Spoof Detection Summary');

saveas(gcf, 'summary_dashboard.png');

fprintf('\n✓ All figures saved:\n');
fprintf('  - confusion_matrix.png\n');
fprintf('  - roc_curve.png\n');
fprintf('  - summary_dashboard.png\n');