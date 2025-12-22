classdef xSTMBGA < ALGORITHM
% <single> <real/integer/label/binary/permutation> <large/none> <constrained/none>
% Genetic algorithm
% proC ---  1 --- Probability of crossover
% disC --- 20 --- Distribution index of simulated binary crossover
% proM ---  1 --- Expectation of the number of mutated variables
% disM --- 20 --- Distribution index of polynomial mutation

%------------------------------- Reference --------------------------------
% Evolutionary Reinforcement Learning with Weight-Freezing and Markov Blanket-Based Dimensionality Reduction,
% 2025.

%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [proC,disC,proM,disM] = Algorithm.ParameterSet(1,20,1,20);
            
            %% Generate random population
            Population = Problem.Initialization();
            count = 0;
            folder = fullfile('D:\sRLXBench\PlatEMO\Data',class(Algorithm)); % change path accordingly
            Archive = Population;
            UMB        = LSMB(Archive,Problem);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                count= count+1;
                MatingPool = TournamentSelection(2,Problem.N,FitnessSingle(Population));
                Offspring  = OperatorGAhalfMB(Problem,Population(MatingPool),UMB);
               
                Population = [Population,Offspring];
                Archive = Population;
                [~,rank]   = sort(FitnessSingle(Population));
                Population = Population(rank(1:Problem.N));
                deepc{count} = UMB;
                if Problem.FE >= Problem.maxFE
                     runNo  = 1;
                     file   = fullfile(folder,sprintf('%s_%s_M%d_D%d_%s',class(Algorithm),class(Problem),Problem.M,Problem.D,'MB'));
                     while exist([file,num2str(runNo),'.mat'],'file') == 2
                        runNo = runNo + 1;
                     end
                    save([file, num2str(runNo),'.mat'],"deepc");
                end


            end
        end
    end

end
