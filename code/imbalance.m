load('ASVspoof_features.mat');
fprintf('Bonafide (0): %d files\n', sum(Y==0));
fprintf('Spoof (1):    %d files\n', sum(Y==1));
fprintf('Ratio: %.2f:1 (Bonafide:Spoof)\n', sum(Y==0)/sum(Y==1));