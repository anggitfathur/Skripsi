% Load your dataset
load('InputNNtrafo3.mat'); % Load the data into MATLAB
trend_component = InputNNtrafo3{:, 4}; % 4th variable

mean_trend_component = mean(trend_component);
std_dev_trend_component = std(trend_component);
standardized_trend = (trend_component - mean_trend_component) / std_dev_trend_component;

numRows = 24;  % Number of rows in 'data'

data = zeros(numRows, 105);  % Initialize the 'data' matrix

for i = 1:numRows
    data(i, :) = standardized_trend(i:i+104)';

end
%split data into input matrix and target matrix
input = data(1:12, :);
target = data(13:24, :);

% Calculate the mean for each column input matrix
means = mean(input, 1);

%transform matrix
input_transform = (input - means);
target_transform = (target - means);

%detransform matrix
input_detransform = input_transform + means;
target_detransform = target_transform + means;

%split training and testing data
trainData = input_transform(:, [1:80, 82:92, 94:104]);
trainTarget = target_transform(:, [1:80, 82:92, 94:104]);
testInput = input_transform(: , [81; 93; 105]);
testTarget = target_transform(: , [81; 93; 105]);

spreadFactors = 0.1:0.1:5; % Range of spread factors to test
MAPE_values = zeros(length(spreadFactors), size(testInput, 2));
MAE_values = zeros(length(spreadFactors), size(testInput, 2));
absolute_percentage_errors_matrix = zeros(length(spreadFactors), size(testInput, 2));
absolute_errors_matrix = zeros(length(spreadFactors), size(testInput, 2));
best_maPe = inf;  % Initialize with a high value
best_net = [];
predicted_values = NaN(50,16);
MAPEtot_values = zeros(length(spreadFactors));
MAEtot_values = zeros(length(spreadFactors));

Z = [];
for i = 1:length(spreadFactors)
    fixedSpreadFactor = spreadFactors(i);

    % Train the GRNN model with the current spread factor
    grnn_model = newgrnn(trainData, trainTarget, fixedSpreadFactor);

    % Predict using the GRNN model with test input
    predicted = sim(grnn_model, testInput);    
    
    testTarget_detransform1 = testTarget(:, 1)+ means(:, 81);
    predicted_detransform1 = predicted(:, 1) + means(:, 81);
    
    testTarget_detransform2 = testTarget(:,2) + means(:, 93);
    predicted_detransform2 = predicted(:, 2) + means(:, 93);

    testTarget_detransform3 = testTarget(:, 3) + means(:, 105);
    predicted_detransform3 = predicted(:, 3) + means(:, 105);
    
    Destandardized_testTarget1 = testTarget_detransform1 * std_dev_trend_component + mean_trend_component; 
    Destandardized_testTarget2 = testTarget_detransform2 * std_dev_trend_component + mean_trend_component;
    Destandardized_testTarget3 = testTarget_detransform3 * std_dev_trend_component + mean_trend_component;
    
    Destandardized_prediction1 = predicted_detransform1 * std_dev_trend_component + mean_trend_component; 
    Destandardized_prediction2 = predicted_detransform2 * std_dev_trend_component + mean_trend_component;
    Destandardized_prediction3 = predicted_detransform3 * std_dev_trend_component + mean_trend_component;
    
    % Calculate sMAPE for the current model
    absolute_percentage_errors1 = abs(Destandardized_testTarget1 - Destandardized_prediction1);
    percentage_errors1 = (absolute_percentage_errors1 ./ abs(Destandardized_testTarget1)) * 100;
    
    absolute_percentage_errors2 = abs(Destandardized_testTarget2 - Destandardized_prediction2);
    percentage_errors2 = (absolute_percentage_errors2 ./ abs(Destandardized_testTarget2)) * 100;
    
    absolute_percentage_errors3 = abs(Destandardized_testTarget3 - Destandardized_prediction3);
    percentage_errors3 = (absolute_percentage_errors3 ./ abs(Destandardized_testTarget3)) * 100;
    
    MAPE = mean([percentage_errors1;percentage_errors2;percentage_errors3]);
    
    % Calculate MAE for the current model
    absolute_errors1 = abs(Destandardized_testTarget1- Destandardized_prediction1);
    absolute_errors2 = abs(Destandardized_testTarget2- Destandardized_prediction2);
    absolute_errors3 = abs(Destandardized_testTarget3- Destandardized_prediction3);
    MAE = mean([absolute_errors1;absolute_errors2;absolute_errors3]);

    % Store the MAPE and MAE values for this spread factor
    MAPE_values(i, :) = MAPE;
    MAE_values(i, :) = MAE;

    %Destandardized data for plotting
    Destandardized_testTarget = [Destandardized_testTarget1',Destandardized_testTarget2',Destandardized_testTarget3'];
    Destandardized_prediction = [Destandardized_prediction1',Destandardized_prediction2',Destandardized_prediction3'];
    
    % Check if this model has a smaller MAE
    if MAPE < best_maPe
    % Update the best MAE and best model
    best_maPe = MAPE;
    best_net = grnn_model;
    Z = predicted;               
    % Close the existing figure and create a new one for the plot
    close(gcf);
    figure;
    plot(Destandardized_testTarget, 'b', 'DisplayName', 'Actual');
    hold on;
    plot(Destandardized_prediction, 'r', 'DisplayName', 'Predictio');
    hold off;
    title(['Actual vs. Predicted Monthly Peak Load (MAPE: ' num2str(MAPE) ')']);
    xlabel('Data Point Index');
    ylabel('Monthly Peak Load');
    legend('show');
    grid on;
    else
        clear grnn_model;                
    end
    
    % Print progress 
    fprintf('Iteration %d/%d - Spread Factor: %.2f, MAPE: %.2f%%, MAE: %.2f\n', i, length(spreadFactors), fixedSpreadFactor, MAPE, MAE);
   
end
% Find the best spread factor with the lowest MAPE
[minMAPE, bestIndexMAPE] = min(MAPE_values);
bestSpreadFactorMAPE = spreadFactors(bestIndexMAPE);
fprintf('Best Spread Parameter (MAPE): %.2f, MAPE: %.2f%%\n ', bestSpreadFactorMAPE, minMAPE);

% Plot MAPE vs. Spread Factor
figure;
plot(spreadFactors, MAPE_values, 'b', 'DisplayName', 'MAPE vs. Spread Parameter');
xlabel('Spread Parameter');
ylabel('Mean Absolute Percentage Error (MAPE)');
title('MAPE vs. Spread Parameter');
grid on;

% Plot MAE values over spread parameter
figure;
plot(spreadFactors, MAE_values, 'g', 'DisplayName', 'MAE vs. Spread Parameter');
xlabel('Spread Parameter');
ylabel('Mean Absolute Error (MAE)');
title('MAE vs. Spread Parameter');
grid on;

%Save Best model
save('GRNN_TSF_Direct_MIMO_trafo3.mat', 'best_net');
