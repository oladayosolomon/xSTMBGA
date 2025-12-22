classdef ProjectedGA < ALGORITHM
% <single> <real> <large/none> 
% Genetic algorithm with intrinsic dimension projection
% projD --- 10 --- Dimensionality of projection space
% updateFreq --- 50 --- Frequency to update projection matrix
% proC ---  1  --- Crossover probability
% disC --- 20  --- SBX distribution index
% proM ---  1  --- Mutation probability
% disM --- 20  --- Polynomial mutation distribution index

%-------------------------- Custom Extension -----------------------------
% Search is done in a lower-dimensional projected space.
% Evaluation is done in the full decision space.
%--------------------------------------------------------------------------
    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [projD, updateFreq, proC, disC, proM, disM] = ...
                Algorithm.ParameterSet(10, 50, 1, 20, 1, 20);

            %% Initialization
            Population = Problem.Initialization();
            D = Problem.D;
            N = Problem.N;
            Gen = 1;

            % Random projection matrix (projD x D)
            P = randn(projD, D) / sqrt(projD);
            P_inv = pinv(P);  % For mapping back

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Selection based on full-dimensional fitness
                MatingPool = TournamentSelection(2, N, FitnessSingle(Population));

                % Get projected decision variables
                ProjDec = zeros(N, projD);
                for i = 1:N
                    ProjDec(i,:) = (P * Population(i).dec')';
                end
                ParentsProj = ProjDec(MatingPool,:);

                % Offspring generation in projected space
                OffProj = OperatorGA_proj(ParentsProj, {proC, disC, proM, disM});

                % Map projected offspring back to full dimension
                OffDec = OffProj * P_inv';
                OffDec = min(max(OffDec, Problem.lower), Problem.upper);

                % Evaluate offspring in full dimension
                Offspring = Problem.Evaluation(OffDec);

                % Environmental selection (mu + lambda)
                Population = [Population, Offspring];
                [~, rank]  = sort(FitnessSingle(Population));
                Population = Population(rank(1:N));

       
            end
        end
    end
end

