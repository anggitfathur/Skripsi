% Load dataset
load('InputNNtrafo4.mat'); % Load the data into MATLAB
Industri = inputNNtrafo4{:, 8}; % 4th variable

%Perhitungan Standarisasi dengan Z-score Standardization
mean_Industri = mean(Industri);
std_dev_Industri = std(Industri);
standardized_Industri = (Industri - mean_Industri) / std_dev_Industri;

%Matrix data Input-Target
numRows = 24;  
numCol = length(standardized_Industri)-numRows+1;

MIMO_input_data = zeros(numRows, 105);  
bestSpreadFactorMAPE = zeros(1,1);

for i = 1:numRows
    MIMO_input_data(i, :) = standardized_Industri(i:i+(numCol-1))';

end
%split ke data Input dan Target
input = MIMO_input_data(1:12, :);
target = MIMO_input_data(13:24, :);

%Nilai rata-rata input 
means = mean(input, 1);

%Transformasi matrix input dan target dengan additive transformation
input_transform = (input - means);
target_transform = (target - means);

%detransform matrix
input_detransform = input_transform + means;
target_detransform = target_transform + means;

%split training dan testing
trainData = input_transform(:, [1:80, 82:92, 94:104]);
trainTarget = target_transform(:, [1:80, 82:92, 94:104]);
testInput = input_transform(: , [81; 93; 105]);
testTarget = target_transform(: , [81; 93; 105]);

%initialisasi matrix yang dibutuhkan
spreadparameter = 0.1:0.1:15; %range spreadparameter
MAPE_values = zeros(length(spreadparameter), size(testInput, 2));
MAE_values = zeros(length(spreadparameter), size(testInput, 2));
absolute_percentage_errors_matrix = zeros(length(spreadparameter), size(testTarget, 1)*size(testTarget, 2));
absolute_errors_matrix = zeros(length(spreadparameter), size(testTarget, 1)*size(testTarget, 2));
best_maPe = inf; 
MIMO_GRNN_Industri_T4_best_net = [];
combinations_matrix = zeros(length(spreadparameter), 3);


for i = 1:length(spreadparameter)
fixedSpreadFactor = spreadparameter(i);

% Training GRNN
grnn_model = newgrnn(trainData, trainTarget, fixedSpreadFactor);

% Testing GRNN
predicted = sim(grnn_model, testInput);    

%De-transformasi target dan output testing
testTarget_detransform1 = testTarget(:, 1)+ means(:, 81);
prediction_detransform1 = predicted(:, 1) + means(:, 81);

testTarget_detransform2 = testTarget(:,2) + means(:, 93);
prediction_detransform2 = predicted(:, 2) + means(:, 93);

testTarget_detransform3 = testTarget(:, 3) + means(:, 105);
prediction_detransform3 = predicted(:, 3) + means(:, 105);

%De-standarisasi target dan output testing
Destandardized_testTarget1 = testTarget_detransform1 * std_dev_Industri + mean_Industri; 
Destandardized_testTarget2 = testTarget_detransform2 * std_dev_Industri + mean_Industri;
Destandardized_testTarget3 = testTarget_detransform3 * std_dev_Industri + mean_Industri;

Destandardized_prediction1 = prediction_detransform1 * std_dev_Industri + mean_Industri; 
Destandardized_prediction2 = prediction_detransform2 * std_dev_Industri + mean_Industri;
Destandardized_prediction3 = prediction_detransform3 * std_dev_Industri + mean_Industri;

%Perhitungan Nilai MAE dan MAPE
absolute_errors1 = abs(Destandardized_testTarget1 - Destandardized_prediction1);
absolute_percentage_errors1 = (absolute_errors1 ./ abs(Destandardized_testTarget1)) * 100;

absolute_errors2 = abs(Destandardized_testTarget2 - Destandardized_prediction2);
absolute_percentage_errors2 = (absolute_errors2 ./ abs(Destandardized_testTarget2)) * 100;

absolute_errors3 = abs(Destandardized_testTarget3 - Destandardized_prediction3);
absolute_percentage_errors3 = (absolute_errors3 ./ abs(Destandardized_testTarget3)) * 100;

absolute_percentage_errors = [absolute_percentage_errors1',absolute_percentage_errors2',absolute_percentage_errors3'];
absolute_errors = [absolute_errors1',absolute_errors2',absolute_errors3'];

MAPE = mean(absolute_percentage_errors);
MAE = mean(absolute_errors);

%Save nilai MAPE dan MAE tiap iterasi
MAPE_values(i, :) = MAPE;
MAE_values(i, :) = MAE;
combinations_matrix(i, 1) = fixedSpreadFactor;
combinations_matrix(i, 2) = MAPE;
combinations_matrix(i, 3) = MAE;
absolute_errors_matrix (i, :) = absolute_errors;
absolute_percentage_errors_matrix (i, :) = absolute_percentage_errors;
if MAPE < best_maPe
% Update best model
best_maPe = MAPE;
MIMO_GRNN_Industri_T4_best_net = grnn_model;

%Penggabungan data untuk plotting
Destandardized_testTarget = [Destandardized_testTarget1',Destandardized_testTarget2',Destandardized_testTarget3'];
Destandardized_prediction = [Destandardized_prediction1',Destandardized_prediction2',Destandardized_prediction3'];

%Plot hasil model terbaik
close(gcf);
figure;
plot(Destandardized_testTarget, 'b', 'DisplayName', 'Target');
hold on;
plot(Destandardized_prediction, 'r', 'DisplayName', 'Prediksi');
hold off;
title(['Target vs Hasil Prediksi (MAPE: ' num2str(MAPE) ')']);
xlabel('Data Point Index');
ylabel('PDRB');
legend('show');
grid on;
clear grnn_model;
Prediction_Industri = Destandardized_prediction;
B_absolute_errors = absolute_errors;
B_absolute_percentage_errors = absolute_percentage_errors;
else
    clear grnn_model;                
end

%Progress
fprintf('Iteration %d/%d - Spread Parameter: %.2f, MAPE: %.2f%%, MAE: %.2f\n', i, length(spreadparameter), fixedSpreadFactor, MAPE, MAE);
   
end
%Penentuan best spread parameter
[min_mape, min_mape_idx] = min(combinations_matrix(:, 2));
best_combination = combinations_matrix(min_mape_idx, :);

fprintf('Best Spread Parameter: %.2f\n ', best_combination(1));
fprintf('MAPE Value: %.4f%%\n ', best_combination(2));
fprintf('MAE Value: %.4f\n ', best_combination(3));

% Plot MAPE vs.over spread parameter
figure;
plot(spreadparameter, MAPE_values, 'b', 'DisplayName', 'MAPE vs. Spread Parameter');
xlabel('Spread Parameter');
ylabel('Mean Absolute Percentage Error (MAPE)');
title('MAPE vs. Spread Parameter');
grid on;

% Plot MAE values over spread parameter
figure;
plot(spreadparameter, MAE_values, 'g', 'DisplayName', 'MAE vs. Spread Parameter');
xlabel('Spread Parameter');
ylabel('Mean Absolute Error (MAE)');
title('MAE vs. Spread Parameter');
grid on;

%Save Best model
save('GRNN_TSF_Industri_Direct_MIMO_trafo4.mat', 'MIMO_GRNN_Industri_T4_best_net');
save('Prediction_Result_Industri_GRNN_T4.mat', 'Prediction_Industri');
save('matrix_data_GRNN_MIMO_Industri_T4.mat', 'MIMO_input_data');
save('best_combination_GRNN_Industri_T4.mat','best_combination'); 
save('combinations_matrix_Industri_T4_GRNN_MIMO.mat.mat','best_combination'); 
save('absolute_errors_matrix_Industri_T4_GRNN_MIMO.mat', 'absolute_errors_matrix')
save('absolute_errors_percentage_matrix_Industri_T4_GRNN_MIMO.mat', 'absolute_percentage_errors_matrix')