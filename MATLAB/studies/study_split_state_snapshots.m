function [X1, X2] = study_split_state_snapshots(X, n_control, num_time_samples)
state_dim = size(X, 1);
X1 = zeros(state_dim, n_control * (num_time_samples - 1));
X2 = zeros(state_dim, n_control * (num_time_samples - 1));

write_idx = 1;
for traj_idx = 1:n_control
    start_idx = (traj_idx - 1) * num_time_samples + 1;
    stop_idx = traj_idx * num_time_samples;
    trajectory = X(:, start_idx:stop_idx);

    snapshot_count = num_time_samples - 1;
    X1(:, write_idx:write_idx + snapshot_count - 1) = trajectory(:, 1:end-1);
    X2(:, write_idx:write_idx + snapshot_count - 1) = trajectory(:, 2:end);
    write_idx = write_idx + snapshot_count;
end
end
