% Load your dataset
load('InputNNtrafo4.mat'); % Load the data into MATLAB
load('GRNN_TSF_Industri_Direct_MIMO_trafo4.mat');

Industri = inputNNtrafo4{:, 8}; % 4th variable

mean_Industri = mean(Industri);
std_dev_Industri = std(Industri);
standardized_Industri = (Industri - mean_Industri) / std_dev_Industri;

numRows = 24;  
numCol = length(standardized_Industri)-numRows+1;

forecast_input_data = zeros(numRows, 105);  
bestSpreadFactorMAPE = zeros(1,1);

for i = 1:numRows
    forecast_input_data(i, :) = standardized_Industri(i:i+(numCol-1))';

end

% Initial data and parameters

DataForecast = forecast_input_data(2:13, 105);
numIterations = 144; % Set the number of iteration

meanForecast_values = zeros(numIterations);
Data_values = zeros(numIterations, 2*size(DataForecast, 1));
Forecast_transform_values = zeros(size(DataForecast, 1),numIterations);
Datanew_values = zeros(size(DataForecast, 1),numIterations);

meanDataForecast_values = [];
DataForecast_values = [];
DataForecast_transform_values = [];
forecast_values = [];
NewDataForecast_values = [];
Destandardized_forecast_values = [];
newData_values = [];

meanForecast = mean(DataForecast);
for i = 1:numIterations
    
    % Standardize the data
    DataForecast_transform = DataForecast - meanForecast;
    
    % Perform the forecast using the neural network
    forecast = sim(MIMO_GRNN_PDRB_T4_best_net, DataForecast_transform); 
    
    % Detransform the forecast
    forecast_detransform = forecast + meanForecast;
    DataForecast_detransform = DataForecast_transform + meanForecast;
    
    newData = [DataForecast_detransform;forecast_detransform];
    
    meanDataForecast_values(i)= meanForecast;
    DataForecast_values (:, i) = DataForecast;
    DataForecast_transform_values (:, i) = DataForecast_transform;
    forecast_values (:, i) = forecast;
    newData_values(:, i) = newData;
    % De-Standardize the forecast if needed
    destandardized_forecast = (forecast_detransform * std_dev_Industri) + mean_Industri;
    Destandardized_forecast_values(:, i) = destandardized_forecast;
    Result_forecast_Industri (:, i) = destandardized_forecast(12); 
    % Display the current iteration
    fprintf('Iteration %d\n', i);
    
    % Display the forecast for this iteration
    fprintf('Forecast: %f\n', destandardized_forecast);
    
    % Prepare the forecast for the next iteration
    DataForecast = newData(2:13,1);
    meanForecast = mean(newData(2:13,1));
end
save('Result_forecast_MIMO_GRNN_Industri_T4.mat', 'Result_forecast_Industri')