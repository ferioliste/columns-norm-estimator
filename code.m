clc, clear all, close all

rng(42);

n = 1000;

A = generate_matrix(n);
a_real = sum(A .* A, 1)';

nnorm = norm(A'*A - diag(a_real), 'fro');
f = @(delta, ell) sqrt(log(2./delta)./ell) .* nnorm;

%% Plot error vs ell

ell_values = unique(floor(logspace(1,3,400)).');
error = zeros(length(ell_values), 1);
for ell_idx = 1:length(ell_values)
    ell = ell_values(ell_idx);
    
    a_est = estimator(A, ell);

    error(ell_idx) = norm(a_real - a_est);
end

figure;
plot(ell_values, error, '-', 'DisplayName', 'Estimator error')    
hold on
plot(ell_values, 0.75./ell_values.^0.5, '-', 'DisplayName', '$c/\sqrt{\ell}$')

xlabel('$\ell$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Error', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 12);
title('Error vs. $\ell$', 'Interpreter', 'latex', 'FontSize', 12);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
print('-depsc2','-vector','error_vs_ell')

%% Plot empirical vs theoretical quantiles when varying ell

ell_values = unique(floor(logspace(1,3,200)).');
delta = 0.05;
times = 30;
sample_quantiles = zeros(length(ell_values), 1);
for ell_idx = 1:length(ell_values)
    ell = ell_values(ell_idx);
    
    error = zeros(times, 1);
    for times_idx = 1:times
        a_est = estimator(A, ell);
        error(times_idx) = norm(a_real - a_est);
    end

    sample_quantiles(ell_idx) = quantile(error,1-delta);
end

figure;
plot(ell_values, sample_quantiles, '-', 'DisplayName', 'Empirical quantiles')    
hold on
plot(ell_values, f(delta, ell_values), '-', 'DisplayName', 'Theoretical quantiles')

xlabel('$\ell$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Quantiles', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 12);
title('Empirical vs theoretical quantiles when varying $\ell$', 'Interpreter', 'latex', 'FontSize', 12);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
print('-depsc2','-vector','quantiles_ell')

%% Plot empirical vs theoretical quantiles when varying delta
% for efficiency we compute 1000 errors and then we compute the empirical quantiles

ell = 750;
deltas = logspace(log10(0.01),log10(0.2),200);
times = 1000;
errors = zeros(times, 1);
sample_quantiles = zeros(length(deltas), 1);

for times_idx = 1:times
    a_est = estimator(A, ell);
    errors(times_idx) = norm(a_real - a_est);
end

for delta_idx = 1:length(deltas)
    delta = deltas(delta_idx);
    sample_quantiles(delta_idx) = quantile(errors,1-delta);
end

figure;
plot(log(2./deltas), sample_quantiles, '-', 'DisplayName', 'Empirical quantiles')    
hold on
plot(log(2./deltas), 0.7*f(deltas, ell), '-', 'DisplayName', 'Theoretical quantiles')

xlabel('$\log(2/\delta)$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Quantiles', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 12);
title('Empirical vs theoretical quantiles when varying $\delta\in[0.01,0.2]$', 'Interpreter', 'latex', 'FontSize', 12);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
print('-depsc2','-vector','quantiles_delta')

%% Functions

function a_est = estimator(A, ell)
    B = A' * A;
    w = rademacher(size(A,2), ell);
    a_est = mean(w .* (B*w),2);
end

function w = rademacher(m, n)
    w = 2 * (rand(m, n) > 0.5) - 1;
end

function A = generate_matrix(n)
    Sigma = diag((1:n).^(-1));
    U = orth(randn(n));
    V = orth(randn(n));
    A = U * Sigma * V';
end