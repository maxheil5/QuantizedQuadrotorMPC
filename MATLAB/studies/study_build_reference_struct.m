function reference = study_build_reference_struct(name, label, X_ref, context)
required_length = context.reference_horizon;
if size(X_ref, 2) < required_length
    error('Reference %s is shorter than the required MPC horizon.', name);
end

reference = struct();
reference.name = string(name);
reference.label = string(label);
reference.X_ref = X_ref(:, 1:required_length);
reference.Z_ref = zeros(size(context.Z0, 1), required_length);
reference.t = (0:(required_length - 1)) * context.params.simTimeStep;

for idx = 1:required_length
    x_ref = reference.X_ref(:, idx);
    basis = get_basis(x_ref, context.n_basis);
    reference.Z_ref(:, idx) = [x_ref(1:3); x_ref(4:6); basis];
end
end
