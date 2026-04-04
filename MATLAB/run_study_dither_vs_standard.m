function results = run_study_dither_vs_standard(opts)
if nargin < 1
    opts = struct();
end

matlab_root = fileparts(mfilename('fullpath'));
addpath(matlab_root);
addpath(fullfile(matlab_root, 'studies'));
study_add_paths(matlab_root);
opts = study_default_options(opts, matlab_root, 'dither_vs_standard_quantization');

if isfield(opts, 'shared_context') && ~isempty(opts.shared_context)
    context = opts.shared_context;
else
    context = study_build_context(opts);
end

bits = [8 12 16];
num_realizations = opts.num_realizations;
total_rows = numel(bits) * (num_realizations + 1);

bit_values = zeros(total_rows, 1);
mode_values = strings(total_rows, 1);
run_indices = zeros(total_rows, 1);
trial_seeds = nan(total_rows, 1);
e_A = zeros(total_rows, 1);
e_B = zeros(total_rows, 1);
e_pred = zeros(total_rows, 1);
e_MPC = zeros(total_rows, 1);

row_idx = 0;
for bit_idx = 1:numel(bits)
    bits_i = bits(bit_idx);
    for run_idx = 1:num_realizations
        model_seed = opts.rng_seed + 1000 * bits_i + run_idx;
        model_result = study_identify_model(context, bits_i, "dithered", model_seed);
        tracking_result = study_evaluate_tracking(context, model_result.EDMD, context.default_reference, false);

        row_idx = row_idx + 1;
        bit_values(row_idx) = bits_i;
        mode_values(row_idx) = "dithered";
        run_indices(row_idx) = run_idx;
        trial_seeds(row_idx) = model_seed;
        e_A(row_idx) = model_result.e_A;
        e_B(row_idx) = model_result.e_B;
        e_pred(row_idx) = model_result.e_pred;
        e_MPC(row_idx) = tracking_result.e_MPC;
    end

    standard_model = study_identify_model(context, bits_i, "standard", NaN);
    standard_tracking = study_evaluate_tracking(context, standard_model.EDMD, context.default_reference, false);

    row_idx = row_idx + 1;
    bit_values(row_idx) = bits_i;
    mode_values(row_idx) = "standard";
    run_indices(row_idx) = 0;
    trial_seeds(row_idx) = NaN;
    e_A(row_idx) = standard_model.e_A;
    e_B(row_idx) = standard_model.e_B;
    e_pred(row_idx) = standard_model.e_pred;
    e_MPC(row_idx) = standard_tracking.e_MPC;
end

trial_table = table(bit_values, mode_values, run_indices, trial_seeds, e_A, e_B, e_pred, e_MPC, ...
    'VariableNames', {'bits', 'mode', 'run_index', 'trial_seed', 'e_A', 'e_B', 'e_pred', 'e_MPC'});

comparison_table = build_comparison_table(trial_table, bits);
figure_path = fullfile(opts.study_output_dir, 'dither_vs_standard_comparison.png');
plot_comparison_figure(trial_table, bits, opts, figure_path);

writetable(trial_table, fullfile(opts.study_output_dir, 'trial_metrics.csv'));
writetable(comparison_table, fullfile(opts.study_output_dir, 'comparison_table.csv'));

results = struct();
results.config = build_config(opts, context, bits);
results.raw_trials = trial_table;
results.summary_tables = struct('comparison_table', comparison_table);
results.figure_paths = struct('comparison', figure_path);
results.output_dir = opts.study_output_dir;

save(fullfile(opts.study_output_dir, 'summary.mat'), 'results');
end

function comparison_table = build_comparison_table(trial_table, bits)
num_bits = numel(bits);
dither_mean_e_A = zeros(num_bits, 1);
dither_std_e_A = zeros(num_bits, 1);
standard_mean_e_A = zeros(num_bits, 1);
standard_std_e_A = zeros(num_bits, 1);

dither_mean_e_B = zeros(num_bits, 1);
dither_std_e_B = zeros(num_bits, 1);
standard_mean_e_B = zeros(num_bits, 1);
standard_std_e_B = zeros(num_bits, 1);

dither_mean_e_pred = zeros(num_bits, 1);
dither_std_e_pred = zeros(num_bits, 1);
standard_mean_e_pred = zeros(num_bits, 1);
standard_std_e_pred = zeros(num_bits, 1);

dither_mean_e_MPC = zeros(num_bits, 1);
dither_std_e_MPC = zeros(num_bits, 1);
standard_mean_e_MPC = zeros(num_bits, 1);
standard_std_e_MPC = zeros(num_bits, 1);

for bit_idx = 1:num_bits
    dither_rows = trial_table(trial_table.bits == bits(bit_idx) & trial_table.mode == "dithered", :);
    standard_rows = trial_table(trial_table.bits == bits(bit_idx) & trial_table.mode == "standard", :);

    dither_mean_e_A(bit_idx) = mean(dither_rows.e_A);
    dither_std_e_A(bit_idx) = std(dither_rows.e_A, 0);
    standard_mean_e_A(bit_idx) = mean(standard_rows.e_A);
    standard_std_e_A(bit_idx) = 0;

    dither_mean_e_B(bit_idx) = mean(dither_rows.e_B);
    dither_std_e_B(bit_idx) = std(dither_rows.e_B, 0);
    standard_mean_e_B(bit_idx) = mean(standard_rows.e_B);
    standard_std_e_B(bit_idx) = 0;

    dither_mean_e_pred(bit_idx) = mean(dither_rows.e_pred);
    dither_std_e_pred(bit_idx) = std(dither_rows.e_pred, 0);
    standard_mean_e_pred(bit_idx) = mean(standard_rows.e_pred);
    standard_std_e_pred(bit_idx) = 0;

    dither_mean_e_MPC(bit_idx) = mean(dither_rows.e_MPC);
    dither_std_e_MPC(bit_idx) = std(dither_rows.e_MPC, 0);
    standard_mean_e_MPC(bit_idx) = mean(standard_rows.e_MPC);
    standard_std_e_MPC(bit_idx) = 0;
end

comparison_table = table(bits(:), ...
    dither_mean_e_A, dither_std_e_A, standard_mean_e_A, standard_std_e_A, ...
    dither_mean_e_B, dither_std_e_B, standard_mean_e_B, standard_std_e_B, ...
    dither_mean_e_pred, dither_std_e_pred, standard_mean_e_pred, standard_std_e_pred, ...
    dither_mean_e_MPC, dither_std_e_MPC, standard_mean_e_MPC, standard_std_e_MPC, ...
    'VariableNames', {'bits', ...
    'dither_mean_e_A', 'dither_std_e_A', 'standard_mean_e_A', 'standard_std_e_A', ...
    'dither_mean_e_B', 'dither_std_e_B', 'standard_mean_e_B', 'standard_std_e_B', ...
    'dither_mean_e_pred', 'dither_std_e_pred', 'standard_mean_e_pred', 'standard_std_e_pred', ...
    'dither_mean_e_MPC', 'dither_std_e_MPC', 'standard_mean_e_MPC', 'standard_std_e_MPC'});
end

function plot_comparison_figure(trial_table, bits, opts, figure_path)
fig = figure('Visible', figure_visibility(opts.show_figures), ...
    'Color', [1 1 1], 'Position', [100 100 1200 850]);
plot_comparison_subplot(subplot(2, 2, 1, 'Parent', fig), trial_table, bits, 'e_A', '$e_A$');
plot_comparison_subplot(subplot(2, 2, 2, 'Parent', fig), trial_table, bits, 'e_B', '$e_B$');
plot_comparison_subplot(subplot(2, 2, 3, 'Parent', fig), trial_table, bits, 'e_pred', '$e_{\mathrm{pred}}$');
plot_comparison_subplot(subplot(2, 2, 4, 'Parent', fig), trial_table, bits, 'e_MPC', '$e_{\mathrm{MPC}}$');

print(fig, figure_path, '-dpng', '-r300');
if ~opts.show_figures
    close(fig);
end
end

function plot_comparison_subplot(ax, trial_table, bits, metric_name, y_label_text)
dither_rows = trial_table(trial_table.mode == "dithered", :);
standard_rows = trial_table(trial_table.mode == "standard", :);
positions = 1:numel(bits);

metric_values = [];
group_values = [];
for bit_idx = 1:numel(bits)
    bit_metric = dither_rows.(metric_name)(dither_rows.bits == bits(bit_idx));
    metric_values = [metric_values; bit_metric];
    group_values = [group_values; repmat(positions(bit_idx), numel(bit_metric), 1)];
end

boxplot(ax, metric_values, group_values, 'Positions', positions, 'Labels', cellstr(string(bits)));
hold(ax, 'on');
standard_values = zeros(numel(bits), 1);
for bit_idx = 1:numel(bits)
    standard_values(bit_idx) = standard_rows.(metric_name)(standard_rows.bits == bits(bit_idx));
end
plot(ax, positions, standard_values, '-d', 'LineWidth', 2, 'MarkerSize', 8, ...
    'Color', [0.85 0.33 0.10], 'MarkerFaceColor', [0.85 0.33 0.10]);
set(ax, 'YScale', 'log');
grid(ax, 'on');
box(ax, 'on');
ax.LineWidth = 1.5;
ax.FontSize = 11;
xlabel(ax, 'Word Length (bits)');
ylabel(ax, y_label_text, 'Interpreter', 'latex');
title(ax, y_label_text, 'Interpreter', 'latex');
legend(ax, {'standard'}, 'Location', 'southwest');
end

function state = figure_visibility(show_figures)
if show_figures
    state = 'on';
else
    state = 'off';
end
end

function config = build_config(opts, context, bits)
config = struct();
config.study_name = opts.study_name;
config.output_root = opts.output_root;
config.study_output_dir = opts.study_output_dir;
config.num_realizations = opts.num_realizations;
config.rng_seed = opts.rng_seed;
config.show_figures = opts.show_figures;
config.show_waitbar = opts.show_waitbar;
config.word_lengths = bits;
config.reference_name = context.default_reference.name;
config.dt = context.dt;
config.t_span = context.t_span;
config.n_control = context.n_control;
config.n_basis = context.n_basis;
config.pred_horizon = context.params.predHorizon;
config.sim_time_step = context.params.simTimeStep;
config.sim_duration = context.params.SimTimeDuration;
end
