function results = run_study_reference_complexity(opts)
if nargin < 1
    opts = struct();
end

matlab_root = fileparts(mfilename('fullpath'));
addpath(matlab_root);
addpath(fullfile(matlab_root, 'studies'));
study_add_paths(matlab_root);
opts = study_default_options(opts, matlab_root, 'reference_trajectory_complexity');

if isfield(opts, 'shared_context') && ~isempty(opts.shared_context)
    context = opts.shared_context;
else
    context = study_build_context(opts);
end

references = context.reference_library;
bits = [Inf 8 12 16];
num_realizations = opts.num_realizations;
num_finite_bits = sum(isfinite(bits));
total_rows = numel(references) * (1 + num_finite_bits * num_realizations);

bit_values = zeros(total_rows, 1);
reference_names = strings(total_rows, 1);
reference_labels = strings(total_rows, 1);
mode_values = strings(total_rows, 1);
run_indices = zeros(total_rows, 1);
trial_seeds = nan(total_rows, 1);
e_A = zeros(total_rows, 1);
e_B = zeros(total_rows, 1);
e_pred = zeros(total_rows, 1);
e_MPC = zeros(total_rows, 1);

row_idx = 0;
baseline_model = study_identify_model(context, Inf, "none", NaN);
for ref_idx = 1:numel(references)
    tracking_result = study_evaluate_tracking(context, baseline_model.EDMD, references(ref_idx), false);
    row_idx = row_idx + 1;
    bit_values(row_idx) = Inf;
    reference_names(row_idx) = references(ref_idx).name;
    reference_labels(row_idx) = references(ref_idx).label;
    mode_values(row_idx) = "none";
    run_indices(row_idx) = 0;
    trial_seeds(row_idx) = NaN;
    e_A(row_idx) = baseline_model.e_A;
    e_B(row_idx) = baseline_model.e_B;
    e_pred(row_idx) = baseline_model.e_pred;
    e_MPC(row_idx) = tracking_result.e_MPC;
end

for bit_idx = 1:numel(bits)
    bits_i = bits(bit_idx);
    if ~isfinite(bits_i)
        continue
    end

    for run_idx = 1:num_realizations
        model_seed = opts.rng_seed + 1000 * bits_i + run_idx;
        model_result = study_identify_model(context, bits_i, "dithered", model_seed);

        for ref_idx = 1:numel(references)
            tracking_result = study_evaluate_tracking(context, model_result.EDMD, references(ref_idx), false);

            row_idx = row_idx + 1;
            bit_values(row_idx) = bits_i;
            reference_names(row_idx) = references(ref_idx).name;
            reference_labels(row_idx) = references(ref_idx).label;
            mode_values(row_idx) = "dithered";
            run_indices(row_idx) = run_idx;
            trial_seeds(row_idx) = model_seed;
            e_A(row_idx) = model_result.e_A;
            e_B(row_idx) = model_result.e_B;
            e_pred(row_idx) = model_result.e_pred;
            e_MPC(row_idx) = tracking_result.e_MPC;
        end
    end
end

trial_table = table(bit_values, reference_names, reference_labels, mode_values, ...
    run_indices, trial_seeds, e_A, e_B, e_pred, e_MPC, ...
    'VariableNames', {'bits', 'reference_name', 'reference_label', 'mode', ...
    'run_index', 'trial_seed', 'e_A', 'e_B', 'e_pred', 'e_MPC'});

summary_table = build_summary_table(trial_table, references, bits);
trajectory_selections = select_representative_runs(trial_table, references, bits);
representative_trajectories = collect_representative_trajectories(context, references, trajectory_selections);

summary_figure_path = fullfile(opts.study_output_dir, 'reference_complexity_tracking.png');
trajectory_figure_path = fullfile(opts.study_output_dir, 'reference_complexity_trajectories.png');
plot_summary_figure(trial_table, references, bits, opts, summary_figure_path);
plot_representative_trajectories(representative_trajectories, references, bits, opts, trajectory_figure_path);

writetable(trial_table, fullfile(opts.study_output_dir, 'trial_metrics.csv'));
writetable(summary_table, fullfile(opts.study_output_dir, 'reference_complexity_summary.csv'));

results = struct();
results.config = build_config(opts, context, bits, references);
results.raw_trials = trial_table;
results.summary_tables = struct('reference_complexity_summary', summary_table);
results.figure_paths = struct( ...
    'tracking_summary', summary_figure_path, ...
    'representative_trajectories', trajectory_figure_path);
results.representative_trajectories = representative_trajectories;
results.output_dir = opts.study_output_dir;

save(fullfile(opts.study_output_dir, 'summary.mat'), 'results');
end

function summary_table = build_summary_table(trial_table, references, bits)
num_rows = numel(references) * numel(bits);
bit_values = zeros(num_rows, 1);
reference_names = strings(num_rows, 1);
reference_labels = strings(num_rows, 1);
num_trials = zeros(num_rows, 1);
median_e_pred = zeros(num_rows, 1);
iqr_e_pred = zeros(num_rows, 1);
median_e_MPC = zeros(num_rows, 1);
iqr_e_MPC = zeros(num_rows, 1);
mean_e_MPC = zeros(num_rows, 1);
std_e_MPC = zeros(num_rows, 1);
regime_label = strings(num_rows, 1);
near_saturation_start_bit = nan(num_rows, 1);

row_idx = 0;
for ref_idx = 1:numel(references)
    ref_mask = trial_table.reference_name == references(ref_idx).name;
    finite_bits = bits(isfinite(bits));
    finite_summary_pred = zeros(numel(finite_bits), 1);
    finite_summary_mpc = zeros(numel(finite_bits), 1);

    for bit_idx = 1:numel(finite_bits)
        bit_mask = ref_mask & trial_table.bits == finite_bits(bit_idx);
        finite_summary_pred(bit_idx) = median(trial_table.e_pred(bit_mask));
        finite_summary_mpc(bit_idx) = median(trial_table.e_MPC(bit_mask));
    end

    [finite_regimes, near_start_bit] = study_compute_regimes(finite_bits(:), finite_summary_pred, finite_summary_mpc);

    for bit_idx = 1:numel(bits)
        row_idx = row_idx + 1;
        rows = trial_table(ref_mask & trial_table.bits == bits(bit_idx), :);

        bit_values(row_idx) = bits(bit_idx);
        reference_names(row_idx) = references(ref_idx).name;
        reference_labels(row_idx) = references(ref_idx).label;
        num_trials(row_idx) = height(rows);
        median_e_pred(row_idx) = median(rows.e_pred);
        iqr_e_pred(row_idx) = iqr(rows.e_pred);
        median_e_MPC(row_idx) = median(rows.e_MPC);
        iqr_e_MPC(row_idx) = iqr(rows.e_MPC);
        mean_e_MPC(row_idx) = mean(rows.e_MPC);
        std_e_MPC(row_idx) = std(rows.e_MPC, 0);
        near_saturation_start_bit(row_idx) = near_start_bit;

        if isfinite(bits(bit_idx))
            regime_label(row_idx) = finite_regimes(finite_bits == bits(bit_idx));
        else
            regime_label(row_idx) = "unquantized";
        end
    end
end

summary_table = table(reference_names, reference_labels, bit_values, num_trials, ...
    median_e_pred, iqr_e_pred, median_e_MPC, iqr_e_MPC, mean_e_MPC, std_e_MPC, ...
    regime_label, near_saturation_start_bit, ...
    'VariableNames', {'reference_name', 'reference_label', 'bits', 'num_trials', ...
    'median_e_pred', 'iqr_e_pred', 'median_e_MPC', 'iqr_e_MPC', ...
    'mean_e_MPC', 'std_e_MPC', 'regime_label', 'near_saturation_start_bit'});
end

function selections = select_representative_runs(trial_table, references, bits)
num_rows = numel(references) * numel(bits);
reference_names = strings(num_rows, 1);
bit_values = zeros(num_rows, 1);
selected_run_index = zeros(num_rows, 1);
selected_trial_seed = nan(num_rows, 1);
selected_e_MPC = zeros(num_rows, 1);
row_idx = 0;

for ref_idx = 1:numel(references)
    ref_mask = trial_table.reference_name == references(ref_idx).name;
    for bit_idx = 1:numel(bits)
        row_idx = row_idx + 1;
        rows = trial_table(ref_mask & trial_table.bits == bits(bit_idx), :);
        median_error = median(rows.e_MPC);
        [~, best_idx] = min(abs(rows.e_MPC - median_error));
        selected_row = rows(best_idx, :);

        reference_names(row_idx) = references(ref_idx).name;
        bit_values(row_idx) = bits(bit_idx);
        selected_run_index(row_idx) = selected_row.run_index;
        selected_trial_seed(row_idx) = selected_row.trial_seed;
        selected_e_MPC(row_idx) = selected_row.e_MPC;
    end
end

selections = table(reference_names, bit_values, selected_run_index, selected_trial_seed, selected_e_MPC, ...
    'VariableNames', {'reference_name', 'bits', 'selected_run_index', 'selected_trial_seed', 'selected_e_MPC'});
end

function representative_trajectories = collect_representative_trajectories(context, references, selections)
num_rows = height(selections);
representative_trajectories = repmat(struct( ...
    'reference_name', "", ...
    'bits', 0, ...
    'run_index', 0, ...
    'trial_seed', NaN, ...
    'e_MPC', NaN, ...
    'X_ref', [], ...
    'X', [], ...
    't', []), num_rows, 1);

for row_idx = 1:num_rows
    selection = selections(row_idx, :);
    reference = find_reference(references, selection.reference_name);

    if isfinite(selection.bits)
        model_result = study_identify_model(context, selection.bits, "dithered", selection.selected_trial_seed);
    else
        model_result = study_identify_model(context, Inf, "none", NaN);
    end

    tracking_result = study_evaluate_tracking(context, model_result.EDMD, reference, true);
    representative_trajectories(row_idx).reference_name = reference.name;
    representative_trajectories(row_idx).bits = selection.bits;
    representative_trajectories(row_idx).run_index = selection.selected_run_index;
    representative_trajectories(row_idx).trial_seed = selection.selected_trial_seed;
    representative_trajectories(row_idx).e_MPC = tracking_result.e_MPC;
    representative_trajectories(row_idx).X_ref = tracking_result.mpc.X_ref;
    representative_trajectories(row_idx).X = tracking_result.mpc.X;
    representative_trajectories(row_idx).t = tracking_result.mpc.t;
end
end

function plot_summary_figure(trial_table, references, bits, opts, figure_path)
fig = figure('Visible', figure_visibility(opts.show_figures), ...
    'Color', [1 1 1], 'Position', [100 100 1400 450]);

for ref_idx = 1:numel(references)
    ax = subplot(1, numel(references), ref_idx, 'Parent', fig);
    ref_rows = trial_table(trial_table.reference_name == references(ref_idx).name, :);
    plot_reference_subplot(ax, ref_rows, bits, references(ref_idx).label);
end

print(fig, figure_path, '-dpng', '-r300');
if ~opts.show_figures
    close(fig);
end
end

function plot_reference_subplot(ax, ref_rows, bits, title_text)
finite_bits = bits(isfinite(bits));
positions = 2:(numel(finite_bits) + 1);
metric_values = [];
group_values = [];
for bit_idx = 1:numel(finite_bits)
    bit_metric = ref_rows.e_MPC(ref_rows.bits == finite_bits(bit_idx));
    metric_values = [metric_values; bit_metric];
    group_values = [group_values; repmat(positions(bit_idx), numel(bit_metric), 1)];
end

boxplot(ax, metric_values, group_values, 'Positions', positions, ...
    'Labels', cellstr(string(finite_bits)));
hold(ax, 'on');
baseline_value = ref_rows.e_MPC(isinf(ref_rows.bits));
scatter(ax, 1, baseline_value, 80, [0.85 0.33 0.10], 'filled');
set(ax, 'YScale', 'log');
grid(ax, 'on');
box(ax, 'on');
ax.LineWidth = 1.5;
ax.FontSize = 11;
ax.XTick = 1:(numel(finite_bits) + 1);
ax.XTickLabel = [{'Inf'}, cellstr(string(finite_bits))];
xlabel(ax, 'Word Length (bits)');
ylabel(ax, '$e_{\mathrm{MPC}}$', 'Interpreter', 'latex');
title(ax, title_text, 'Interpreter', 'none');
end

function plot_representative_trajectories(representative_trajectories, references, bits, opts, figure_path)
fig = figure('Visible', figure_visibility(opts.show_figures), ...
    'Color', [1 1 1], 'Position', [100 100 1400 900]);
tiledlayout(numel(references), numel(bits), 'Padding', 'compact', 'TileSpacing', 'compact');

for ref_idx = 1:numel(references)
    for bit_idx = 1:numel(bits)
        ax = nexttile;
        trajectory = find_trajectory(representative_trajectories, references(ref_idx).name, bits(bit_idx));

        plot3(ax, trajectory.X_ref(:, 1), trajectory.X_ref(:, 2), trajectory.X_ref(:, 3), '--', 'LineWidth', 2);
        hold(ax, 'on');
        plot3(ax, trajectory.X(:, 1), trajectory.X(:, 2), trajectory.X(:, 3), 'LineWidth', 1.75);
        grid(ax, 'on');
        box(ax, 'on');
        axis(ax, 'equal');
        ax.LineWidth = 1.25;
        ax.FontSize = 10;
        xlabel(ax, 'x');
        ylabel(ax, 'y');
        zlabel(ax, 'z');
        if isfinite(bits(bit_idx))
            title(ax, sprintf('%s | %d bits', references(ref_idx).label, bits(bit_idx)), 'Interpreter', 'none');
        else
            title(ax, sprintf('%s | Inf', references(ref_idx).label), 'Interpreter', 'none');
        end
    end
end

print(fig, figure_path, '-dpng', '-r300');
if ~opts.show_figures
    close(fig);
end
end

function reference = find_reference(references, reference_name)
for ref_idx = 1:numel(references)
    if references(ref_idx).name == reference_name
        reference = references(ref_idx);
        return
    end
end
error('Reference %s not found.', reference_name);
end

function trajectory = find_trajectory(representative_trajectories, reference_name, bits)
for idx = 1:numel(representative_trajectories)
    if representative_trajectories(idx).reference_name == reference_name && ...
            representative_trajectories(idx).bits == bits
        trajectory = representative_trajectories(idx);
        return
    end
end
error('Representative trajectory not found.');
end

function state = figure_visibility(show_figures)
if show_figures
    state = 'on';
else
    state = 'off';
end
end

function config = build_config(opts, context, bits, references)
config = struct();
config.study_name = opts.study_name;
config.output_root = opts.output_root;
config.study_output_dir = opts.study_output_dir;
config.num_realizations = opts.num_realizations;
config.rng_seed = opts.rng_seed;
config.show_figures = opts.show_figures;
config.show_waitbar = opts.show_waitbar;
config.word_lengths = bits;
config.reference_names = {references.name};
config.dt = context.dt;
config.t_span = context.t_span;
config.n_control = context.n_control;
config.n_basis = context.n_basis;
config.pred_horizon = context.params.predHorizon;
config.sim_time_step = context.params.simTimeStep;
config.sim_duration = context.params.SimTimeDuration;
end
