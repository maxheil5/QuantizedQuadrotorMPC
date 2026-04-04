function results = run_study_fine_word_length_sweep(opts)
if nargin < 1
    opts = struct();
end

matlab_root = fileparts(mfilename('fullpath'));
addpath(matlab_root);
addpath(fullfile(matlab_root, 'studies'));
study_add_paths(matlab_root);
opts = study_default_options(opts, matlab_root, 'fine_word_length_sweep');

if isfield(opts, 'shared_context') && ~isempty(opts.shared_context)
    context = opts.shared_context;
else
    context = study_build_context(opts);
end

bits = 4:16;
num_realizations = opts.num_realizations;
total_rows = numel(bits) * num_realizations + 1;

bit_values = zeros(total_rows, 1);
mode_values = strings(total_rows, 1);
reference_values = strings(total_rows, 1);
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
        reference_values(row_idx) = context.default_reference.name;
        run_indices(row_idx) = run_idx;
        trial_seeds(row_idx) = model_seed;
        e_A(row_idx) = model_result.e_A;
        e_B(row_idx) = model_result.e_B;
        e_pred(row_idx) = model_result.e_pred;
        e_MPC(row_idx) = tracking_result.e_MPC;
    end
end

baseline_model = study_identify_model(context, Inf, "none", NaN);
baseline_tracking = study_evaluate_tracking(context, baseline_model.EDMD, context.default_reference, false);

row_idx = row_idx + 1;
bit_values(row_idx) = Inf;
mode_values(row_idx) = "none";
reference_values(row_idx) = context.default_reference.name;
run_indices(row_idx) = 0;
trial_seeds(row_idx) = NaN;
e_A(row_idx) = baseline_model.e_A;
e_B(row_idx) = baseline_model.e_B;
e_pred(row_idx) = baseline_model.e_pred;
e_MPC(row_idx) = baseline_tracking.e_MPC;

trial_table = table(bit_values, mode_values, reference_values, run_indices, trial_seeds, ...
    e_A, e_B, e_pred, e_MPC, ...
    'VariableNames', {'bits', 'mode', 'reference_name', 'run_index', 'trial_seed', ...
    'e_A', 'e_B', 'e_pred', 'e_MPC'});

summary_table = build_summary_table(trial_table, [bits, Inf]);
[regime_detail_table, regime_summary_table] = build_regime_tables(summary_table);

metric_figure_path = fullfile(opts.study_output_dir, 'fine_word_length_metrics.png');
regime_figure_path = fullfile(opts.study_output_dir, 'fine_word_length_regimes.png');
plot_metric_grid(trial_table, [bits, Inf], opts, metric_figure_path);
plot_regime_figure(regime_detail_table, opts, regime_figure_path);

writetable(trial_table, fullfile(opts.study_output_dir, 'trial_metrics.csv'));
writetable(summary_table, fullfile(opts.study_output_dir, 'summary_by_bit.csv'));
writetable(regime_summary_table, fullfile(opts.study_output_dir, 'regime_summary.csv'));

results = struct();
results.config = build_config(opts, context, bits);
results.raw_trials = trial_table;
results.summary_tables = struct( ...
    'summary_by_bit', summary_table, ...
    'regime_detail', regime_detail_table, ...
    'regime_summary', regime_summary_table);
results.figure_paths = struct( ...
    'metric_grid', metric_figure_path, ...
    'regime_summary', regime_figure_path);
results.output_dir = opts.study_output_dir;

save(fullfile(opts.study_output_dir, 'summary.mat'), 'results');
end

function summary_table = build_summary_table(trial_table, bit_list)
num_bits = numel(bit_list);
mode_values = strings(num_bits, 1);
num_trials = zeros(num_bits, 1);
median_e_A = zeros(num_bits, 1);
iqr_e_A = zeros(num_bits, 1);
mean_e_A = zeros(num_bits, 1);
std_e_A = zeros(num_bits, 1);
median_e_B = zeros(num_bits, 1);
iqr_e_B = zeros(num_bits, 1);
mean_e_B = zeros(num_bits, 1);
std_e_B = zeros(num_bits, 1);
median_e_pred = zeros(num_bits, 1);
iqr_e_pred = zeros(num_bits, 1);
mean_e_pred = zeros(num_bits, 1);
std_e_pred = zeros(num_bits, 1);
median_e_MPC = zeros(num_bits, 1);
iqr_e_MPC = zeros(num_bits, 1);
mean_e_MPC = zeros(num_bits, 1);
std_e_MPC = zeros(num_bits, 1);

for bit_idx = 1:num_bits
    rows = trial_table(trial_table.bits == bit_list(bit_idx), :);
    mode_values(bit_idx) = rows.mode(1);
    num_trials(bit_idx) = height(rows);

    median_e_A(bit_idx) = median(rows.e_A);
    iqr_e_A(bit_idx) = iqr(rows.e_A);
    mean_e_A(bit_idx) = mean(rows.e_A);
    std_e_A(bit_idx) = std(rows.e_A, 0);

    median_e_B(bit_idx) = median(rows.e_B);
    iqr_e_B(bit_idx) = iqr(rows.e_B);
    mean_e_B(bit_idx) = mean(rows.e_B);
    std_e_B(bit_idx) = std(rows.e_B, 0);

    median_e_pred(bit_idx) = median(rows.e_pred);
    iqr_e_pred(bit_idx) = iqr(rows.e_pred);
    mean_e_pred(bit_idx) = mean(rows.e_pred);
    std_e_pred(bit_idx) = std(rows.e_pred, 0);

    median_e_MPC(bit_idx) = median(rows.e_MPC);
    iqr_e_MPC(bit_idx) = iqr(rows.e_MPC);
    mean_e_MPC(bit_idx) = mean(rows.e_MPC);
    std_e_MPC(bit_idx) = std(rows.e_MPC, 0);
end

summary_table = table(bit_list(:), mode_values, num_trials, ...
    median_e_A, iqr_e_A, mean_e_A, std_e_A, ...
    median_e_B, iqr_e_B, mean_e_B, std_e_B, ...
    median_e_pred, iqr_e_pred, mean_e_pred, std_e_pred, ...
    median_e_MPC, iqr_e_MPC, mean_e_MPC, std_e_MPC, ...
    'VariableNames', {'bits', 'mode', 'num_trials', ...
    'median_e_A', 'iqr_e_A', 'mean_e_A', 'std_e_A', ...
    'median_e_B', 'iqr_e_B', 'mean_e_B', 'std_e_B', ...
    'median_e_pred', 'iqr_e_pred', 'mean_e_pred', 'std_e_pred', ...
    'median_e_MPC', 'iqr_e_MPC', 'mean_e_MPC', 'std_e_MPC'});
end

function [detail_table, regime_summary_table] = build_regime_tables(summary_table)
finite_summary = summary_table(isfinite(summary_table.bits), :);
[regime_labels, near_start_bit] = study_compute_regimes( ...
    finite_summary.bits, finite_summary.median_e_pred, finite_summary.median_e_MPC);

detail_table = finite_summary(:, {'bits', 'median_e_pred', 'median_e_MPC'});
detail_table.regime_label = regime_labels;
detail_table.near_saturation_start_bit = repmat(near_start_bit, height(detail_table), 1);

regime_names = ["coarse"; "transition"; "near-saturation"];
bit_ranges = strings(numel(regime_names), 1);
bit_values = strings(numel(regime_names), 1);
criteria = [ ...
    "median e_pred or median e_MPC is at least 2x the near-saturation baseline"; ...
    "between coarse and near-saturation"; ...
    "smallest bit where all higher-bit medians stay within 10% of the 16-bit medians"];

for regime_idx = 1:numel(regime_names)
    selected_bits = finite_summary.bits(regime_labels == regime_names(regime_idx));
    if isempty(selected_bits)
        bit_ranges(regime_idx) = "none";
        bit_values(regime_idx) = "[]";
    else
        bit_ranges(regime_idx) = sprintf('%d-%d', selected_bits(1), selected_bits(end));
        bit_values(regime_idx) = join(string(selected_bits(:)).', ' ');
    end
end

regime_summary_table = table(regime_names, bit_ranges, bit_values, ...
    repmat(near_start_bit, numel(regime_names), 1), criteria, ...
    'VariableNames', {'regime_label', 'bit_range', 'bit_values', ...
    'near_saturation_start_bit', 'selection_rule'});
end

function plot_metric_grid(trial_table, bit_list, opts, figure_path)
labels = bit_labels(bit_list);
fig = figure('Visible', figure_visibility(opts.show_figures), ...
    'Color', [1 1 1], 'Position', [100 100 1200 850]);
plot_metric_subplot(subplot(2, 2, 1, 'Parent', fig), trial_table, bit_list, 'e_A', '$e_A$', labels);
plot_metric_subplot(subplot(2, 2, 2, 'Parent', fig), trial_table, bit_list, 'e_B', '$e_B$', labels);
plot_metric_subplot(subplot(2, 2, 3, 'Parent', fig), trial_table, bit_list, 'e_pred', '$e_{\mathrm{pred}}$', labels);
plot_metric_subplot(subplot(2, 2, 4, 'Parent', fig), trial_table, bit_list, 'e_MPC', '$e_{\mathrm{MPC}}$', labels);

print(fig, figure_path, '-dpng', '-r300');
if ~opts.show_figures
    close(fig);
end
end

function plot_metric_subplot(ax, trial_table, bit_list, metric_name, y_label_text, labels)
metric_values = [];
group_values = [];
for bit_idx = 1:numel(bit_list)
    bit_metric = trial_table.(metric_name)(trial_table.bits == bit_list(bit_idx));
    metric_values = [metric_values; bit_metric];
    group_values = [group_values; repmat(bit_idx, numel(bit_metric), 1)];
end

boxplot(ax, metric_values, group_values, 'Labels', cellstr(labels));
set(ax, 'YScale', 'log');
grid(ax, 'on');
box(ax, 'on');
ax.LineWidth = 1.5;
ax.FontSize = 11;
title(ax, y_label_text, 'Interpreter', 'latex');
xlabel(ax, 'Word Length (bits)');
ylabel(ax, y_label_text, 'Interpreter', 'latex');
end

function plot_regime_figure(regime_detail_table, opts, figure_path)
fig = figure('Visible', figure_visibility(opts.show_figures), ...
    'Color', [1 1 1], 'Position', [100 100 1100 700]);
ax1 = subplot(2, 1, 1, 'Parent', fig);
hold(ax1, 'on');
patch(ax1, [8 14 14 8], [1e-12 1e-12 1e3 1e3], [0.92 0.95 1.00], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(ax1, regime_detail_table.bits, regime_detail_table.median_e_pred, '-o', 'LineWidth', 2);
plot(ax1, regime_detail_table.bits, regime_detail_table.median_e_MPC, '-s', 'LineWidth', 2);
set(ax1, 'YScale', 'log');
grid(ax1, 'on');
box(ax1, 'on');
ax1.LineWidth = 1.5;
ax1.FontSize = 11;
xlabel(ax1, 'Word Length (bits)');
ylabel(ax1, 'Median Error');
legend(ax1, {'8-14 highlight', '$e_{\mathrm{pred}}$', '$e_{\mathrm{MPC}}$'}, ...
    'Interpreter', 'latex', 'Location', 'southwest');
title(ax1, 'Transition Trend Across the Fine Word-Length Sweep');

ax2 = subplot(2, 1, 2, 'Parent', fig);
scatter(ax2, regime_detail_table.bits, ones(height(regime_detail_table), 1), 120, ...
    regime_colors(regime_detail_table.regime_label), 'filled');
grid(ax2, 'on');
box(ax2, 'on');
ax2.LineWidth = 1.5;
ax2.FontSize = 11;
ax2.YTick = [];
ylim(ax2, [0.5 1.5]);
xlabel(ax2, 'Word Length (bits)');
title(ax2, 'Regime Classification');
for row_idx = 1:height(regime_detail_table)
    text(ax2, regime_detail_table.bits(row_idx), 1.08, regime_detail_table.regime_label(row_idx), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

print(fig, figure_path, '-dpng', '-r300');
if ~opts.show_figures
    close(fig);
end
end

function colors = regime_colors(regime_labels)
colors = zeros(numel(regime_labels), 3);
for idx = 1:numel(regime_labels)
    switch regime_labels(idx)
        case "coarse"
            colors(idx, :) = [0.85 0.33 0.10];
        case "transition"
            colors(idx, :) = [0.93 0.69 0.13];
        otherwise
            colors(idx, :) = [0.00 0.45 0.74];
    end
end
end

function labels = bit_labels(bit_list)
labels = strings(size(bit_list));
for bit_idx = 1:numel(bit_list)
    if isfinite(bit_list(bit_idx))
        labels(bit_idx) = sprintf('%d', bit_list(bit_idx));
    else
        labels(bit_idx) = "Inf";
    end
end
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
config.unquantized_label = Inf;
config.reference_name = context.default_reference.name;
config.dt = context.dt;
config.t_span = context.t_span;
config.n_control = context.n_control;
config.n_basis = context.n_basis;
config.pred_horizon = context.params.predHorizon;
config.sim_time_step = context.params.simTimeStep;
config.sim_duration = context.params.SimTimeDuration;
end
