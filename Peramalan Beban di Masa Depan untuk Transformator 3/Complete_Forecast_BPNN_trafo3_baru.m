load('InputNNtrafo3.mat');
load('FA_BPNN_T3(2).mat')
load('Result_forecast_BPNN_Trend_T3')

trend_component = InputNNtrafo3{:, 4};
monthly_peak_load = InputNNtrafo3{:, 3};

mean_trend_component = mean(trend_component);
std_dev_trend_component = std(trend_component);
mean_monthly_peak_load = mean(monthly_peak_load);
std_dev_monthly_peak_load = std(monthly_peak_load);

standardized_forecast_trend = (Result_forecast - mean_trend_component) / std_dev_trend_component;

Data_standardized_forecast_trend = reshape(standardized_forecast_trend, 12, 17);

data_input = Data_standardized_forecast_trend;

means = mean(data_input, 1);
input_transform = (data_input - means);

prediction_monthly_peak_load = FA_BPNN_T3_best_net(input_transform);

prediction_monthly_peak_load_detransform = prediction_monthly_peak_load + means;

Destandardized_prediction = prediction_monthly_peak_load_detransform* std_dev_monthly_peak_load + mean_monthly_peak_load;

save('Prediction_Beban_T3.mat', 'Destandardized_prediction' );
