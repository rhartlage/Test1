function [loadings, scores, eigenvalues] = factor_analysis(X, num_factors)
%FACTOR_ANALYSIS Perform a basic factor analysis using PCA.
%   [loadings, scores, eigenvalues] = factor_analysis(X, num_factors)
%   takes a data matrix X (observations x variables) and returns
%   factor loadings, factor scores, and the selected eigenvalues
%   for the specified number of factors.
%
%   Inputs:
%       X           - Data matrix (observations x variables).
%       num_factors - Number of factors to retain.
%
%   Outputs:
%       loadings    - Factor loading matrix (variables x factors).
%       scores      - Factor scores (observations x factors).
%       eigenvalues - Eigenvalues for the retained factors.

    if nargin < 2
        error('factor_analysis requires X and num_factors inputs.');
    end

    [num_obs, num_vars] = size(X);
    if num_factors <= 0 || num_factors > num_vars
        error('num_factors must be between 1 and the number of variables.');
    end

    % Center the data
    mean_X = mean(X, 1);
    X_centered = X - mean_X;

    % Compute covariance matrix
    covariance_matrix = (X_centered' * X_centered) / (num_obs - 1);

    % Eigen decomposition
    [eig_vectors, eig_values_matrix] = eig(covariance_matrix, 'vector');

    % Sort eigenvalues and eigenvectors in descending order
    [eig_values_sorted, sort_idx] = sort(eig_values_matrix, 'descend');
    eig_vectors_sorted = eig_vectors(:, sort_idx);

    % Retain the specified number of factors
    eigenvalues = eig_values_sorted(1:num_factors);
    factors = eig_vectors_sorted(:, 1:num_factors);

    % Compute loadings (variables x factors)
    loadings = factors * diag(sqrt(eigenvalues));

    % Compute factor scores (observations x factors)
    scores = X_centered * factors;
end
