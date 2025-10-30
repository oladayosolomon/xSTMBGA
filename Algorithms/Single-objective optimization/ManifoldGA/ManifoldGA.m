classdef ManifoldGA < ALGORITHM
% <single> <real/integer/label/binary/permutation> <large/none> <constrained/none>
% Genetic algorithm with nonlinear manifold-based projection (ISOMAP)
% projD --- 10 --- Dimensionality of projected manifold
% proC ---  1  --- Crossover probability
% disC --- 20  --- SBX distribution index
% proM ---  1  --- Mutation probability
% disM --- 20  --- Polynomial mutation distribution index

%-------------------------------------------------------------------------
% PCA-based search: Uses PCA to learn a submanifold of the decision space.
% Offspring are generated in the low-dim PCA space and mapped back to full D.
% Fitness is always evaluated in original space.
%-------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [projD, proC, disC, proM, disM] = ...
                Algorithm.ParameterSet(10, 1, 20, 1, 20);

            %% Initialization
            Population = Problem.Initialization();
            D = Problem.D;
            N = Problem.N;
            Gen = 1;

            %% Initial PCA projection matrix
            X = cat(1, Population.dec);             % (N x D)
            [coeff, ~, ~, ~, ~, mu] = pca(X);       % PCA components and mean
            P = coeff(:,1:projD);                   % (D x projD)
            mu = mu(:)';                            % (1 x D)

            %% Optimization loop
            while Algorithm.NotTerminated(Population)
                % Tournament selection in full space
                MatingPool = TournamentSelection(2, N, FitnessSingle(Population));

                % Project selected parents to PCA space
                ParentDec = cat(1, Population(MatingPool).dec);   % (N x D)
                ProjParents = (ParentDec - mu) * P;               % (N x projD)

                % Crossover + mutation in PCA space
                OffProj = OperatorGA_proj(ProjParents, {proC, disC, proM, disM});

                % Map offspring back to full D using inverse PCA transform
                OffDec = OffProj * P' + mu;                       % (N x D)
                OffDec = min(max(OffDec, Problem.lower), Problem.upper);

                % Evaluate in original space
                Offspring = Problem.Evaluation(OffDec);

                % Environmental selection (μ + λ)
                Population = [Population, Offspring];
                [~, rank] = sort(FitnessSingle(Population));
                Population = Population(rank(1:N));

               
            end
        end
    end
end
