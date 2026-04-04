function results = run_all_new_thesis_studies(opts)
if nargin < 1
    opts = struct();
end

matlab_root = fileparts(mfilename('fullpath'));
addpath(matlab_root);
addpath(fullfile(matlab_root, 'studies'));
study_add_paths(matlab_root);

shared_opts = study_default_options(opts, matlab_root, 'all_new_thesis_studies');
shared_context = study_build_context(shared_opts);
opts.shared_context = shared_context;

results = struct();
results.fine_word_length_sweep = run_study_fine_word_length_sweep(opts);
results.dither_vs_standard = run_study_dither_vs_standard(opts);
results.reference_complexity = run_study_reference_complexity(opts);
results.output_root = shared_opts.output_root;
end
