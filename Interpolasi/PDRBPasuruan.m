% Import Data
load('PDRBPasuruan.mat');
x = PDRBPasuruan(:, 1);
y = PDRBPasuruan(:, 2);

% Membuat Spline Interpolation
splineInterpolant = csapi(x, y);

% Define Nilai x yang dicari
x_values_to_evaluate = [1:156];

% Membuat Interpolasi di tiap nilai x
y_interpolated_float = fnval(splineInterpolant, x_values_to_evaluate);

% Membuat vektor bulan-tahun untuk plotting
date_values = datetime('2010-01', 'Format', 'yyyy-MM'):calmonths(1):datetime('2022-12', 'Format', 'yyyy-MM');
initial_date = date_values(1);

% Mengubah vektor x menjadi format bulan-tahun
x_based_date = initial_date + calmonths(x- 1);

% Plot Hasil interpolasi dengan nilai y 
plot(date_values, y_interpolated_float, 'k-', x_based_date, y, 'ro');
title('Interpolasi Data PDRB Pasuruan');

grid on

% Membuat Label x-axis
xlabel('Bulan-Tahun'); % Label for the x-axis
ylabel('Triliun Rupiah');  % Label for the y-axis

xtickformat('MMM-yyyy');

% Set Legend untuk plot
legend('Interpolasi', 'PDRB')
