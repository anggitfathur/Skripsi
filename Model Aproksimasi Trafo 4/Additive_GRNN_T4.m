a% Load dataset
load('InputNNtrafo4.mat'); % Load the data into MATLAB
load('Prediction_Result_Trend_Beban_GRNN_T4.mat');
load('Prediction_Result_PDRB_GRNN_T4.mat');
load('Prediction_Result_Industri_GRNN_T4.mat');

trend_component = inputNNtrafo4{:, 4}; % 4th variable
monthly_peak_load = inputNNtrafo4{:, 3}; % 3rd variable
PDRB = inputNNtrafo4{:, 7}; % 3rd variable
Industri = inputNNtrafo4{:, 8}; % 3rd variable

new_trend_component = [trend_component(1:92);Prediction_Beban'];
new_Industri = [Industri(1:92);Prediction_Industri'];
new_PDRB = [PDRB(1:92);Prediction_PDRB'];

mean_trend_component = mean(trend_component);
std_dev_trend_component = std(trend_component);
standardized_trend = (new_trend_component - mean_trend_component) / std_dev_trend_component;

mean_PDRB = mean(PDRB);
std_dev_PDRB = std(PDRB);
standardized_PDRB = (new_PDRB - mean_PDRB) / std_dev_PDRB;

mean_Industri = mean(Industri);
std_dev_Industri = std(Industri);
standardized_Industri = (new_Industri - mean_Industri) / std_dev_Industri;

mean_monthly_peak_load = mean(monthly_peak_load);
std_dev_monthly_peak_load = std(monthly_peak_load);
standardized_monthly_peak_load = (monthly_peak_load - mean_monthly_peak_load) / std_dev_monthly_peak_load;

numRows = 12;  % Number of rows in 'data'
numCol = length(standardized_trend)-numRows+1;

data_trend = zeros(numRows, numCol);  % Initialize the 'data' matrix
bestSpreadFactorMAPE = zeros(1,1);
data_monthly_peak_load = zeros(numRows, numCol);
data_Industri = zeros(numRows, numCol);
data_PDRB = zeros(numRows, numCol);
for a = 1:numRows
    data_trend(a, :) = standardized_trend(a:a+(numCol-1))';
    data_monthly_peak_load(a, :) = standardized_monthly_peak_load(a:a+(numCol-1))';
    data_Industri (a, :) = standardized_Industri(a:a+(numCol-1))';
    data_PDRB (a, :) = standardized_PDRB(a:a+(numCol-1))';
end

%split ke data Input dan Target
input = [data_trend;data_PDRB;data_Industri];
target = data_monthly_peak_load;

%Nilai rata-rata input 
means = mean(input, 1);

%Transformasi matrix input dan target dengan additive transformation
input_transform = (input - means);
target_transform = (target - means);

%detransform matrix
input_detransform = input_transform + means;
target_detransform = target_transform + means;

%split training dan testing
trainData = input_transform(:, [1:92, 93:104, 106:116]);
trainTarget = target_transform(:, [1:92, 93:104, 106:116]);
testInput = input_transform(: , [93; 105; 117]);
testTarget = target_transform(: , [93; 105; 117]);

spreadparameter = 0.01:0.01:1.5; %range spreadparameter
MAPE_values = zeros(length(spreadparameter), size(testInput, 2));
MAE_values = zeros(length(spreadparameter), size(testInput, 2));
absolute_percentage_errors_matrix = zeros(length(spreadparameter), size(testTarget, 1)*size(testTarget, 2));
absolute_errors_matrix = zeros(length(spreadparameter), size(testTarget, 1)*size(testTarget, 2));
best_maPe = inf;
combinations_matrix = zeros(length(spreadparameter), 3);

Z = [];
for i = 1:length(spreadparameter)
    fixedSpreadFactor = spreadparameter(i);

    % Training GRNN
    grnn_model = newgrnn(trainData, trainTarget, fixedSpreadFactor);

    % Testing GRNN
    predicted = sim(grnn_model, testInput);    
    
    %De-transformasi target dan output testing
    testTarget_detransform1 = testTarget(:, 1)+ means(:, 93);
    predicted_detransform1 = predicted(:, 1) + means(:, 93);
    
    testTarget_detransform2 = testTarget(:,2) + means(:, 105);
    predicted_detransform2 = predicted(:, 2) + means(:, 105);

    testTarget_detransform3 = testTarget(:, 3) + means(:, 117);
    predicted_detransform3 = predicted(:, 3) + means(:, 117);
    
    %De-standarisasi target dan output testing
    Destandardized_testTarget1 = testTarget_detransform1 * std_dev_monthly_peak_load + mean_monthly_peak_load; 
    Destandardized_testTarget2 = testTarget_detransform2 * std_dev_monthly_peak_load + mean_monthly_peak_load;
    Destandardized_testTarget3 = testTarget_detransform3 * std_dev_monthly_peak_load + mean_monthly_peak_load;

    Destandardized_prediction1 = predicted_detransform1 * std_dev_monthly_peak_load + mean_monthly_peak_load; 
    Destandardized_prediction2 = predicted_detransform2 * std_dev_monthly_peak_load + mean_monthly_peak_load;
    Destandardized_prediction3 = predicted_detransform3 * std_dev_monthly_peak_load + mean_monthly_peak_load;
    
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
    FA_GRNN_T4_best_net = grnn_model;
    
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
    title(['Target vs. Prediksi (MAPE: ' num2str(MAPE) ')']);
    xlabel('Data Point Index');
    ylabel('Beban Puncak Bulanan');
    legend('show');
    grid on;
    Prediction = Destandardized_prediction;
    save('FA_GRNN_T4.mat', 'FA_GRNN_T4_best_net');
    B_absolute_errors = absolute_errors;
    B_absolute_percentage_errors = absolute_percentage_errors;
    clear grnn_model;
    else
        clear grnn_model;                
    end
    
    % Print progress 
    fprintf('Iteration %d/%d - Spread Parameter: %.2f, MAPE: %.2f%%, MAE: %.2f\n', i, length(spreadparameter), fixedSpreadFactor, MAPE, MAE);
   
end
% Find the best spread factor with the lowest MAPE
[min_mape, min_mape_idx] = min(combinations_matrix(:, 2));
best_combination = combinations_matrix(min_mape_idx, :);

fprintf('Best Spread Parameter: %.2f\n ', best_combination(1));
fprintf('MAPE Value: %.4f%%\n ', best_combination(2));
fprintf('MAE Value: %.4f\n ', best_combination(3));

% Plot MAPE vs. Spread Factor
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
save('Prediction_FA_GRNN_T4.mat', 'Prediction');
save('best_combination_FA_GRNN_T4.mat', 'best_combination'); 
save('combinations_matrix_FA_GRNN_T4.mat', 'combinations_matrix')
save('absolute_errors_matrix_FA_GRNN_T4.mat', 'absolute_errors_matrix')
save('absolute_errors_matrix_FA_GRNN_T4.mat', 'absolute_errors_matrix')