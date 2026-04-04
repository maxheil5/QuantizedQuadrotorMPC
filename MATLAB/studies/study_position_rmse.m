function e_mpc = study_position_rmse(X, X_ref)
error_signal = X_ref - X;
numerator = mean(sum(error_signal.^2, 2));
denominator = mean(sum(X_ref.^2, 2));
e_mpc = sqrt(numerator / max(denominator, eps));
end
