function diversity = computeDiversity(pop)
% Computes diversity of each solution in the population
% Inputs:
%   pop   - an NxM matrix: N = number of individuals, M = dimensions (objectives or decision variables)
%   space - string: 'decision' or 'objective' to indicate space used
%
% Output:
%   diversity - Nx1 vector with diversity score per individual

    if nargin < 2
        space = 'objective'; % Default to objective space
    end

    N = size(pop, 1);  % Number of individuals
    diversity = zeros(N, 1);

    % Compute pairwise distances
    distMatrix = squareform(pdist(pop));  % Euclidean distances

    for i = 1:N
        % Average distance to all other solutions
        diversity(i) = mean(distMatrix(i, [1:i-1, i+1:N]));
    end

    % Optional: Normalize
    %diversity = diversity / max(diversity);
end
