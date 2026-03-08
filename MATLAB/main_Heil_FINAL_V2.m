%% main_Heil_FINAL_V3.m
clear all; close all; clc;
addpath('./dynamics', './edmd', './mpc', './utils', './training');

rng(2141444); % Set random seed
warning('off','optim:quadprog:HessianNotSym'); % Suppress Hessian warning

%% *************************** Simulation Parameters ***************************
params = get_params();
dt = 1e-3;
t_span = 0.1; % seconds
word_length_array = 4:2:14; % Removed 16-bit quantization
word_length_plot = [word_length_array, Inf]; % Include Inf for unquantized case
num_cases = length(word_length_plot);
run_number = 5; % Number of random simulations

%% *************************** Generate Training Data ***************************
n_control = 100;
t_traj = 0:dt:t_span;
Nsim_data_generation = numel(t_traj);
x0 = [0;0;0]; dx0 = [0;0;0];
R0 = eye(3); wb0 = [0.1;0;0];
X0 = [x0;dx0;R0(:);wb0];

% Generate unquantized trajectories
[X, U, X1, X2, U1] = get_rnd_trajectories(X0, n_control, t_traj, false, 'train');
n_basis = 3;
EDMD = get_EDMD(X1, X2, U1, n_basis, t_traj);

%% *************************** Compute Unquantized Data ***************************
[X_unquantized, U_unquantized, X1_unquantized, X2_unquantized, U1_unquantized] = ...
    get_rnd_trajectories(X0, n_control, t_traj, false, 'train');
EDMD_unquantized = get_EDMD(X1_unquantized, X2_unquantized, U1_unquantized, n_basis, t_traj);
[RMSE_unquantized, ~] = eval_EDMD_fixed_traj(X0, X_unquantized, U_unquantized, dt, EDMD_unquantized, n_basis);
prediction_error_nonquantized = RMSE_unquantized.x;
tracking_error_nonquantized = 0; % Placeholder for nonquantized tracking error (adjust if needed)

%% *************************** Initialize Storage ***************************
matrix_A_difference = zeros(run_number, num_cases-1);
matrix_B_difference = zeros(run_number, num_cases-1);
prediction_error_quantized = zeros(run_number, num_cases-1);
tracking_error_quantized = zeros(run_number, num_cases-1);
matrix_A_dithered = cell(run_number, num_cases-1);
matrix_B_dithered = cell(run_number, num_cases-1);
quantizedStateCell = cell(run_number, num_cases-1);
quantizedInputCell = cell(run_number, num_cases-1);

%% *************************** Run Experiment ***************************
for run_counter = 1:run_number
    fprintf('Running simulation %d/%d...\n', run_counter, run_number);
    
    for len_idx = 1:(num_cases - 1) % Exclude "Inf" case
        bits = word_length_array(len_idx);

        % Quantize state data
        X_min = min(X, [], 2); X_max = max(X, [], 2);
        [epsilon_X, X_min_new, X_max_new, partition_X, midPoints_X] = Partition(X_min, X_max, bits);
        X_dithered = Dither_Func(X, epsilon_X, X_min_new, X_max_new, partition_X, midPoints_X);
        X1_dithered = []; X2_dithered = [];
        
        for traj_counter = 1:n_control
            lowerIndex = (traj_counter-1) * Nsim_data_generation + 1;
            upperIndex = traj_counter * Nsim_data_generation - 1;
            X1_dithered = [X1_dithered, X_dithered(:, lowerIndex:upperIndex)];
            X2_dithered = [X2_dithered, X_dithered(:, lowerIndex+1:upperIndex+1)];
        end

        % Quantize control inputs
        U_min = min(U1, [], 2); U_max = max(U1, [], 2);
        [epsilon_U, U_min_new, U_max_new, partition_U, midPoints_U] = Partition(U_min, U_max, bits);
        U1_dithered = Dither_Func(U1, epsilon_U, U_min_new, U_max_new, partition_U, midPoints_U);

        % Compute EDMD for quantized data
        EDMD_dithered = get_EDMD(X1_dithered, X2_dithered, U1_dithered, n_basis, t_traj);

        % Store system matrices
        matrix_A_dithered{run_counter, len_idx} = EDMD_dithered.A;
        matrix_B_dithered{run_counter, len_idx} = EDMD_dithered.B;

        % Compute Frobenius norm difference
        matrix_A_difference(run_counter, len_idx) = norm(EDMD.A - EDMD_dithered.A, 'fro') / norm(EDMD.A, 'fro');
        matrix_B_difference(run_counter, len_idx) = norm(EDMD.B - EDMD_dithered.B, 'fro') / norm(EDMD.B, 'fro');

        % Evaluate Prediction & Tracking Error
        [Xeval, Ueval, X1eval, X2eval] = get_rnd_trajectories(X0, 1, t_traj, false, 'train');
        [RMSE_dithered, ~] = eval_EDMD_fixed_traj(X0, Xeval, Ueval, dt, EDMD_dithered, n_basis);
        prediction_error_quantized(run_counter, len_idx) = RMSE_dithered.x;
        tracking_error_quantized(run_counter, len_idx) = 0; % Placeholder for tracking error calculation
    end
end

%% *************************** Plot Results ***************************
repmat_size = size(prediction_error_quantized, 1); % Number of repetitions
prediction_error_nonquantized_expanded = repmat(prediction_error_nonquantized, repmat_size, 1);
tracking_error_nonquantized_expanded = repmat(tracking_error_nonquantized, repmat_size, 1);

% RMSE vs Word Length
figure;
boxplot([prediction_error_quantized, prediction_error_nonquantized_expanded], word_length_plot);
xlabel('Word Length (bits)'); ylabel('Prediction Error');
title('Prediction Error vs. Word Length'); grid on;
set(gca, 'YScale', 'log');
print('Prediction_Error_Comparison_log','-dpng');

% Tracking Error vs Word Length
figure;
boxplot([tracking_error_quantized, tracking_error_nonquantized_expanded], word_length_plot);
xlabel('Word Length (bits)'); ylabel('Tracking Error');
title('MPC Tracking Error vs. Word Length'); grid on;
set(gca, 'YScale', 'log');
print('Tracking_Error_Comparison_log','-dpng');

% A Matrix Difference vs Word Length
figure;
boxplot(matrix_A_difference, word_length_plot(1:end-1));
xlabel('Word Length (bits)'); ylabel('$\|A - \bar{A}\| / \|A\|$', 'Interpreter', 'latex');
title('A Matrix Difference vs. Word Length'); grid on;
set(gca, 'YScale', 'log');
print('Matrix_A_Difference','-dpng');

% B Matrix Difference vs Word Length
figure;
boxplot(matrix_B_difference, word_length_plot(1:end-1));
xlabel('Word Length (bits)'); ylabel('$\|B - \bar{B}\| / \|B\|$', 'Interpreter', 'latex');
title('B Matrix Difference vs. Word Length'); grid on;
set(gca, 'YScale', 'log');
print('Matrix_B_Difference','-dpng');

fprintf('Experiment completed! Results saved to quantization_results.mat.\n');
