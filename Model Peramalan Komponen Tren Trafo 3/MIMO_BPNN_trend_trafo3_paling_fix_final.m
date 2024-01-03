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

% Define hyperparameter search ranges
learning_rates = (0.01:0.05:0.11);  % Adjust the learning rates to your needs
momentums = (0.6:0.07:0.9);  % Adjust the momentum values to your needs
neuron = (1:1:10);  % Adjust the hidden layer sizes to your needs
epochs = (1000:100:2000);  % Adjust the epoch values to your needs

% Calculate the total number of combinations
total_combinations = length(learning_rates) * length(momentums) * length(neuron) * length(epochs);

% Determine the maximum number of iterations
max_iterations = total_combinations;

% Preallocate matrices and arrays
combinations_matrix = zeros(max_iterations, 6);

mae_matrix = zeros(max_iterations, 1);
mape_matrix = zeros(max_iterations, 1);

% Preallocate arrays for other variables
best_mae = inf;  % Initialize with a high value
MIMO_BPNN_T3_best_net = [];
progress_counter = 0;
progress_percentage = 0;
first_plot = true;

% Initialize an index variable to keep track of the current row in the matrices
matrix_index = 0;
best_mape = inf;  % Initialize with a high value
absolute_errors_matrix = zeros(max_iterations, size(testTarget, 1)*size(testTarget, 2));
absolute_percentage_errors_matrix = zeros(max_iterations, size(testTarget, 1)*size(testTarget, 2));

% Nested loops for hyperparameter search
for lr = learning_rates
    for mc = momentums
        for nr = neuron
            for ep = epochs
% Create a feedforward neural network
net = feedforwardnet(nr, 'traingdm');

% Set common BPNN parameters
net.trainParam.epochs = ep;
net.trainParam.lr = lr;
net.trainParam.mc = mc;
net.trainParam.goal = 0.01;
net.trainParam.min_grad = 1e-6;

% Train the neural network
[net, tr] = train(net, trainData, trainTarget);

% Test the neural network
predicted = net(testInput);

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

% Calculate MAPE for the current model
absolute_errors1 = abs(Destandardized_testTarget1 - Destandardized_prediction1);
absolute_percentage_errors1 = (absolute_errors1 ./ abs(Destandardized_testTarget1)) * 100;

absolute_errors2 = abs(Destandardized_testTarget2 - Destandardized_prediction2);
absolute_percentage_errors2 = (absolute_errors2 ./ abs(Destandardized_testTarget2)) * 100;

absolute_errors3 = abs(Destandardized_testTarget3 - Destandardized_prediction3);
absolute_percentage_errors3 = (absolute_errors3 ./ abs(Destandardized_testTarget3)) * 100;

absolute_percentage_errors = [absolute_percentage_errors1',absolute_percentage_errors2',absolute_percentage_errors3'];
absolute_errors = [absolute_errors1',absolute_errors2',absolute_errors3'];

MAPE = mean(absolute_percentage_errors);

% Calculate MAE for the current model
MAE = mean(absolute_errors);

% Store the MAPE and MAE values for this spread factor
MAPE_values(i, :) = MAPE;
MAE_values(i, :) = MAE;

%Destandardized data for plotting
Destandardized_testTarget = [Destandardized_testTarget1',Destandardized_testTarget2',Destandardized_testTarget3'];
Destandardized_prediction = [Destandardized_prediction1',Destandardized_prediction2',Destandardized_prediction3'];

% Store this combination and corresponding MAE in the matrix
combination = [lr, mc, nr, ep, MAE, MAPE];
% Inside the loop, when you want to add data to the matrices
matrix_index = matrix_index + 1;
combinations_matrix(matrix_index, :) = combination;
mape_matrix(matrix_index) = MAPE;
mae_matrix(matrix_index) = MAE;

% Check if this model has a smaller MAE
if MAPE < best_mape
    % Update the best MAE and best model
    best_mape = MAPE;
    best_net_baru = net;

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
Prediction = [Destandardized_prediction1',Destandardized_prediction2', Destandardized_prediction3'];
B_combination = [lr, mc, nr, ep];
B_absolute_errors = absolute_errors;
B_absolute_percentage_errors = absolute_percentage_errors;
clear net;
else 
    clear net;
end

% Save the best model
save('BPNN_TSF_Direct_MIMO_trafo3(a).mat', 'best_net_baru');

% Update progress
progress_counter = progress_counter + 1;
progress_percentage = (progress_counter / total_combinations) * 100;
fprintf('Progress: %.5f%%\n', progress_percentage);
fprintf('Best MAPE: %.4f\n', best_mape);
fprintf('Test MSE: %.4f\n', MAE);
fprintf('Test MAPE: %.2f%%\n', MAPE);
fprintf('Best combination: %.2f\n', B_combination);
                
            end
        end
    end
end

% Display the best hyperparameters and corresponding MAE
[min_mae, min_mape_idx] = min(combinations_matrix(:, 6));
best_combination = combinations_matrix(min_mape_idx, :);

% Display the best hyperparameters and corresponding MAE
fprintf('Best Hyperparameters:\n');
fprintf('Learning Rate: %.2f\n', best_combination(1));
fprintf('Momentum: %.2f\n', best_combination(2));
fprintf('Hidden Layer Size: %d\n', best_combination(3));
fprintf('Epochs: %d\n', best_combination(4));
fprintf('MAE: %d\n', best_combination(5));
fprintf('MAPE: %d\n', best_combination(6));

%save('input_data_BPNN_trafo3.mat', 'Prediction'); 
save('Prediction_Result_Trend_BPNN_T3(a).mat', 'Prediction')
save('best_combination_MIMO_BPNN_trafo3(a).mat', 'best_combination'); 
save('combinations_matrix_T3_BPNN_MIMO(a).mat', 'combinations_matrix')
save('absolute_errors_matrix_T3_BPNN_MIMO(a).mat', 'absolute_errors_matrix')
save('absolute_errors_percentage_matrix_T3_BPNN_MIMO(a).mat', 'absolute_percentage_errors_matrix')