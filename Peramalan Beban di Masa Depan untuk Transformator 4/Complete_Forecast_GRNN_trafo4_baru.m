load('InputNNtrafo4.mat');
load('FA_BPNN_T4.mat')
load('Result_forecast_MIMO_GRNN_Trend_Beban_T4.mat')
load('Result_forecast_MIMO_GRNN_Industri_T4.mat')
load('Result_forecast_MIMO_GRNN_PDRB_T4.mat')

trend_component = inputNNtrafo4{:, 4};
monthly_peak_load = inputNNtrafo4{:, 3};
Industri = inputNNtrafo4{:, 8};
PDRB = inputNNtrafo4{:, 7};

mean_trend_component = mean(trend_component);
std_dev_trend_component = std(trend_component);

mean_monthly_peak_load = mean(monthly_peak_load);
std_dev_monthly_peak_load = std(monthly_peak_load);

mean_Industri = mean(Industri);
std_dev_Industri = std(Industri);

mean_PDRB = mean(PDRB);
std_dev_PDRB = std(PDRB);

standardized_forecast_trend = (Result_forecast_trend - mean_trend_component) / std_dev_trend_component;
standardized_forecast_PDRB = (Result_forecast_PDRB - mean_PDRB) / std_dev_PDRB;
standardized_forecast_Industri = (Result_forecast_Industri - mean_Industri) / std_dev_Industri;
data_input = [reshape(standardized_forecast_trend, 12, 12);reshape(standardized_forecast_PDRB, 12, 12);reshape(standardized_forecast_Industri, 12, 12)]; 

means = mean(data_input, 1);
input_transform = (data_input - means);

prediction_monthly_peak_load = sim(FA_GRNN_T4_best_net, input_transform); 

prediction_monthly_peak_load_detransform = prediction_monthly_peak_load + means;

Destandardized_prediction = prediction_monthly_peak_load_detransform* std_dev_monthly_peak_load + mean_monthly_peak_load;

save('Prediction_Beban_T4.mat', 'Destandardized_prediction')