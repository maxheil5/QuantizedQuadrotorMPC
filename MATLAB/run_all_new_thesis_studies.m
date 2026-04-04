function results = run_all_new_thesis_studies(opts)
if nargin < 1
    opts = struct();
end

if ~isfield(opts, 'show_progress') || isempty(opts.show_progress)
    opts.show_progress = true;
end

matlab_root = fileparts(mfilename('fullpath'));
addpath(matlab_root);
addpath(fullfile(matlab_root, 'studies'));
study_add_paths(matlab_root);

total_steps = 3;
run_timer = tic;
if opts.show_progress
    fprintf('Running all new thesis studies...\n');
    print_progress(0, total_steps, 'Building shared study context', run_timer);
end

shared_opts = study_default_options(opts, matlab_root, 'all_new_thesis_studies');
shared_context = study_build_context(shared_opts);
opts.shared_context = shared_context;

results = struct();
if opts.show_progress
    print_progress(0, total_steps, 'Running Fine Word-Length Sweep', run_timer);
end
results.fine_word_length_sweep = run_study_fine_word_length_sweep(opts);
if opts.show_progress
    print_progress(1, total_steps, 'Completed Fine Word-Length Sweep', run_timer);
    print_progress(1, total_steps, 'Running Dithered and Standard Comparison', run_timer);
end
results.dither_vs_standard = run_study_dither_vs_standard(opts);
if opts.show_progress
    print_progress(2, total_steps, 'Completed Dithered and Standard Comparison', run_timer);
    print_progress(2, total_steps, 'Running Reference Trajectory Complexity', run_timer);
end
results.reference_complexity = run_study_reference_complexity(opts);
results.output_root = shared_opts.output_root;

if opts.show_progress
    print_progress(3, total_steps, 'Completed all new thesis studies', run_timer);
end
end

function print_progress(step_idx, total_steps, message, run_timer)
bar_width = 30;
progress = step_idx / total_steps;
filled_width = round(progress * bar_width);
bar_text = [repmat('#', 1, filled_width), repmat('-', 1, bar_width - filled_width)];
elapsed_seconds = toc(run_timer);
fprintf('[%s] %3.0f%% %s (elapsed %.1fs)\n', ...
    bar_text, 100 * progress, message, elapsed_seconds);
end
