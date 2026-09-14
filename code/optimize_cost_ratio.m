%% OPTIMIZE COST RATIO - Reduce False Alarms
clear; clc; close all;

%% Load data
load('combined_features.mat');  % X_combined, Y
load('test_indices.mat');       % testIdx, trainIdx

%% Prepare data
[X_train, mu, sigma] = zscore(X_combined(trainIdx, :));
X_test = (X_combined(testIdx, :) - mu) ./ sigma;

%% Test different cost ratios
cost_ratios = [1, 2, 3, 5, 8.8, 10, 12, 15, 20, 30, 50];
results = [];

fprintf('========================================\n');
fprintf('OPTIMIZING COST RATIO (False Spoof Penalty)\n');
fprintf('========================================\n');
fprintf('Ratio | Accuracy | Precision | Recall | F1-Score | FP | FN\n');
fprintf('------|----------|-----------|--------|----------|----|----\n');

for i = 1:length(cost_ratios)
    ratio = cost_ratios(i);
    cost_mat = [0, ratio; 1, 0];
    
    svm_temp = fitcsvm(X_train, Y(trainIdx), ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', 1, ...
        'Cost', cost_mat);
    
    pred_temp = predict(svm_temp, X_test);
    
    % Calculate metrics
    tp = sum(pred_temp == 1 & Y(testIdx) == 1);
    tn = sum(pred_temp == 0 & Y(testIdx) == 0);
    fp = sum(pred_temp == 1 & Y(testIdx) == 0);
    fn = sum(pred_temp == 0 & Y(testIdx) == 1);
    
    acc = (tp + tn) / length(testIdx);
    prec = tp / (tp + fp + eps);
    rec = tp / (tp + fn + eps);
    f1 = 2 * prec * rec / (prec + rec + eps);
    
    results(i).ratio = ratio;
    results(i).accuracy = acc;
    results(i).precision = prec;
    results(i).recall = rec;
    results(i).f1 = f1;
    results(i).fp = fp;
    results(i).fn = fn;
    
    fprintf('%5.1f  | %7.2f%% | %8.2f%% | %6.2f%% | %7.2f%% | %3d | %3d\n', ...
        ratio, acc*100, prec*100, rec*100, f1*100, fp, fn);
end

%% Find optimal based on different criteria
% Criterion 1: Maximize F1-Score
[~, best_f1_idx] = max([results.f1]);
best_f1_ratio = results(best_f1_idx).ratio;

% Criterion 2: Minimize False Positives (FP) while keeping Recall > 95%
valid_idx = [results.recall] >= 0.95;
if any(valid_idx)
    valid_fp = [results(valid_idx).fp];
    [~, min_fp_idx] = min(valid_fp);
    temp_idx = find(valid_idx);
    best_fp_ratio = results(temp_idx(min_fp_idx)).ratio;
else
    best_fp_ratio = best_f1_ratio;
end

fprintf('\n========================================\n');
fprintf('OPTIMAL COST RATIOS:\n');
fprintf('========================================\n');
fprintf('Max F1-Score:       ratio = %.1f (F1 = %.2f%%)\n', ...
    best_f1_ratio, results(best_f1_idx).f1*100);
fprintf('Min False Positives: ratio = %.1f (FP = %d, Recall = %.1f%%)\n', ...
    best_fp_ratio, results(find([results.ratio]==best_fp_ratio)).fp, ...
    results(find([results.ratio]==best_fp_ratio)).recall*100);

%% Visualization
figure('Position', [100, 100, 1000, 600]);

% Plot 1: Metrics vs Cost Ratio
subplot(2,2,1);
plot(cost_ratios, [results.accuracy]*100, 'b-o', 'LineWidth', 1.5);
hold on;
plot(cost_ratios, [results.precision]*100, 'g-s', 'LineWidth', 1.5);
plot(cost_ratios, [results.recall]*100, 'r-^', 'LineWidth', 1.5);
plot(cost_ratios, [results.f1]*100, 'm-d', 'LineWidth', 1.5);
xlabel('Cost Ratio (False Spoof Penalty)');
ylabel('Percentage (%)');
title('Performance vs Cost Ratio');
legend('Accuracy', 'Precision', 'Recall', 'F1-Score', 'Location', 'best');
grid on;

% Plot 2: False Positives and False Negatives
subplot(2,2,2);
yyaxis left;
plot(cost_ratios, [results.fp], 'r-o', 'LineWidth', 1.5);
ylabel('False Positives (Bonafide → Spoof)');
yyaxis right;
plot(cost_ratios, [results.fn], 'b-s', 'LineWidth', 1.5);
ylabel('False Negatives (Spoof → Bonafide)');
xlabel('Cost Ratio');
title('Classification Errors vs Cost Ratio');
legend('False Positives', 'False Negatives', 'Location', 'best');
grid on;

% Plot 3: Recommended operating point
subplot(2,2,3);
plot([results.recall]*100, [results.precision]*100, 'bo-', 'LineWidth', 1.5);
hold on;
plot(results(best_f1_idx).recall*100, results(best_f1_idx).precision*100, ...
    'rs', 'MarkerSize', 12, 'LineWidth', 2);
plot(results(find([results.ratio]==best_fp_ratio)).recall*100, ...
    results(find([results.ratio]==best_fp_ratio)).precision*100, ...
    'gd', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('Recall (%)'); ylabel('Precision (%)');
title('Precision-Recall Trade-off');
legend('All Ratios', 'Max F1', 'Min FP', 'Location', 'best');
grid on;

% Plot 4: Bar chart comparison
subplot(2,2,4);
comparison_ratios = [best_f1_ratio, best_fp_ratio];
comparison_f1 = [results(best_f1_idx).f1*100, results(find([results.ratio]==best_fp_ratio)).f1*100];
comparison_fp = [results(best_f1_idx).fp, results(find([results.ratio]==best_fp_ratio)).fp];

bar_data = [comparison_f1; comparison_fp ./ max(comparison_fp) * 100];
bar(bar_data');
set(gca, 'XTickLabel', {'Max F1', 'Min FP'});
ylabel('Score (%)');
title(sprintf('Comparison: Ratio %.1f vs %.1f', best_f1_ratio, best_fp_ratio));
legend('F1-Score', 'FP (normalized)', 'Location', 'best');
grid on;

sgtitle('Cost Ratio Optimization for Ear-Witness');

saveas(gcf, 'cost_optimization.png');

%% Recommendation
fprintf('\n========================================\n');
fprintf('RECOMMENDATION\n');
fprintf('========================================\n');

if best_fp_ratio ~= best_f1_ratio
    fprintf('For your project:\n');
    fprintf('  - If you CANNOT tolerate false alarms: Use ratio = %.1f\n', best_fp_ratio);
    fprintf('    (FP = %d, but F1 drops to %.1f%%)\n', ...
        results(find([results.ratio]==best_fp_ratio)).fp, ...
        results(find([results.ratio]==best_fp_ratio)).f1*100);
    fprintf('\n  - If you want balanced performance: Use ratio = %.1f (Max F1)\n', best_f1_ratio);
    fprintf('    (F1 = %.1f%%, FP = %d)\n', results(best_f1_idx).f1*100, results(best_f1_idx).fp);
else
    fprintf('Optimal ratio: %.1f (F1 = %.1f%%, FP = %d)\n', best_f1_ratio, ...
        results(best_f1_idx).f1*100, results(best_f1_idx).fp);
end

%% Save optimal model (optional)
% Uncomment to retrain with optimal ratio
% optimal_ratio = best_fp_ratio;  % or best_f1_ratio
% cost_mat_opt = [0, optimal_ratio; 1, 0];
% svm_optimal = fitcsvm(X_train, Y(trainIdx), ...
%     'KernelFunction', 'rbf', ...
%     'Cost', cost_mat_opt);
% save('svm_optimal_model.mat', 'svm_optimal', 'mu', 'sigma');