function opts = study_default_options(opts, matlab_root, study_name)
if nargin < 1 || isempty(opts)
    opts = struct();
end

opts = set_default(opts, 'output_root', fullfile(matlab_root, 'results'));
opts = set_default(opts, 'num_realizations', 50);
opts = set_default(opts, 'rng_seed', 2141444);
opts = set_default(opts, 'show_figures', false);
opts = set_default(opts, 'show_waitbar', false);
opts = set_default(opts, 'study_name', study_name);

opts.study_output_dir = fullfile(opts.output_root, opts.study_name);
if ~exist(opts.output_root, 'dir')
    mkdir(opts.output_root);
end
if ~exist(opts.study_output_dir, 'dir')
    mkdir(opts.study_output_dir);
end
end

function opts = set_default(opts, field_name, default_value)
if ~isfield(opts, field_name) || isempty(opts.(field_name))
    opts.(field_name) = default_value;
end
end
