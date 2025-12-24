classdef xSTMBCMAES < ALGORITHM
% <single> <real/integer/label/binary/permutation> <large/none> <constrained/none>
% Parent-wise CMA-ES with Markov Blanket MB, parent-wise frozen inheritance,
% and per-parent adaptive scalar step-sizes (sigma_p).
%
% Simple sigma adaptation: multiplicative update using a success indicator.

    methods
        function main(Algorithm, Problem)
            %% Parameters
            p_target = 0.2;           % 1/5 success rule target
            c_scale  = 0.2;           % base learning rate, scaled by dim
            soft_perturb = 0.0;       % optional small noise on frozen dims (keep 0 or small)
            min_sigma = 1e-8;
            max_sigma = 1e2;

            %% CMA-ES basics (only for MB covariance learning)
            mu     = round(Problem.N/2);
            w      = log(mu + 0.5) - log(1:mu); w = w ./ sum(w);
            mu_eff = 1 / sum(w.^2);
            cs  = (mu_eff + 2) / (Problem.D + mu_eff + 5);
            ds  = 1 + cs + 2 * max(sqrt((mu_eff - 1) / (Problem.D + 1)) - 1, 0);
            ENN = sqrt(Problem.D) * (1 - 1/(4*Problem.D) + 1/(21*Problem.D^2));
            cc  = (4 + mu_eff/Problem.D) / (4 + Problem.D + 2*mu_eff/Problem.D);
            c1  = 2 / ((Problem.D + 1.3)^2 + mu_eff);
            cmu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((Problem.D + 2)^2 + 2*mu_eff/2));
            hth = (1.4 + 2/(Problem.D + 1)) * ENN;

            %% Initial population
            Population = Problem.Initialization(Problem.N);
            if isa(Population, 'SOLUTION')
                PopDec = reshape([Population.decs], Problem.D, [])';
            else
                PopDec = Population;
            end

            % compute initial mean (for bookkeeping)
            Mdec = mean(PopDec, 1);

            %% Markov Blanket (MB) selection - indices
            UMB = LSMB(Population, Problem);
            if isempty(UMB); UMB = 1:Problem.D; end
            D_MB = numel(UMB);

            %% MB-subspace CMA-ES state
            C_MB = eye(D_MB);
            ps_MB = zeros(1, D_MB);
            pc_MB = zeros(1, D_MB);
            sigma_MB = 0.1 * mean(Problem.upper - Problem.lower); % baseline scalar

            %% Per-parent sigma initialization
            sigma_p = sigma_MB * ones(Problem.N, 1);  % one scalar per parent

            %% Logging
            deepc = {}; gen = 0; folder = fullfile('D:\sRLXBench\PlatEMO\Data', class(Algorithm));

            %% Main loop
            while Algorithm.NotTerminated(Population)
                gen = gen + 1;

                % extract decisions if SOLUTION struct
                if isa(Population, 'SOLUTION')
                    PopDec = reshape([Population.decs], Problem.D, [])';
                else
                    PopDec = Population;
                end
                N = size(PopDec,1);

                % Ensure C_MB symmetric PSD
                C_MB = (C_MB + C_MB')/2;
                [V,E] = eig(C_MB); E = diag(E); E(E<=0) = 1e-12; C_MB = V*diag(E)*V';

                % Each parent produces one offspring by mutating only MB dims
                Pdec_off = PopDec;  % initialize offspring as copies of parents
                Pstep_MB = zeros(N, D_MB);
                for i = 1:N
                    % sample MB step using MB covariance, then scale by parent sigma
                    try
                        z = mvnrnd(zeros(1,D_MB), C_MB, 1);   % [1 x D_MB]
                    catch
                        z = randn(1,D_MB);
                    end
                    step = sigma_p(i) * z;                  % per-parent scalar scaling
                    Pdec_off(i, UMB) = PopDec(i, UMB) + step;
                    Pstep_MB(i, :) = step / max(sigma_p(i), 1e-20); % normalized step for updates
                    % optional small perturbation to frozen dims
                    if soft_perturb > 0
                        frozenIdx = setdiff(1:Problem.D, UMB);
                        Pdec_off(i, frozenIdx) = Pdec_off(i, frozenIdx) + soft_perturb * sigma_p(i) * randn(1, numel(frozenIdx));
                    end
                end

                % Evaluate offspring
                Offspring = Problem.Evaluation(Pdec_off);

                % Combine parents and offspring (μ+λ) and select best N individuals
                Combined = [Population, Offspring]; % expected as SOLUTION objects or matrices
                % compute fitness and ranks
                fitness_all = FitnessSingle(Combined);
                [~, idx_sorted] = sort(fitness_all);
                survivors = idx_sorted(1:Problem.N);
                % Build next generation Population_decisions
                % If SOLUTION objects, rebuild; otherwise work with matrices
                if isa(Combined, 'SOLUTION')
                    Population = Combined(survivors);
                    PopDec = reshape([Population.decs], Problem.D, [])';
                else
                    PopDec = zeros(Problem.N, Problem.D);
                    for k=1:Problem.N
                        PopDec(k,:) = Combined(survivors(k), :);
                    end
                    Population = PopDec;
                end

                % Determine which offspring were selected (success indicators)
                % For each parent i, find index of its offspring in Combined (parents are indices 1:N, offspring N+1:2N)
                selected_flags = zeros(N,1); % s_i =1 if offspring i survived
                for i = 1:N
                    offspring_idx_in_combined = N + i; % 1-based position
                    if any(survivors == offspring_idx_in_combined)
                        selected_flags(i) = 1;
                    end
                end

                % Update per-parent sigma using simple multiplicative rule
                % learning rate scaled by MB dimension
                c = c_scale / sqrt(max(D_MB,1));
                for i=1:N
                    s_i = selected_flags(i);
                    sigma_p(i) = sigma_p(i) * exp( (s_i - p_target) * c );
                    % clamp sigma to reasonable range
                    sigma_p(i) = min(max_sigma, max(min_sigma, sigma_p(i)));
                end

                % --- Update MB-subspace CMA-ES parameters using selected top mu individuals
                % We need MB steps of selected individuals (or approximate using surviving population)
                % Use top mu survivors' MB coordinates to compute weighted Mstep
                if isa(Population, 'SOLUTION')
                    PopDec = reshape([Population.decs], Problem.D, [])';
                end
                topMB = PopDec(1:mu, UMB); % top mu in current population
                % compute steps relative to previous Mdec_MB estimate:
                % we maintain Mdec_MB as the last MB mean (recompute from previous if lacking)
                % For simplicity, set old MB mean to mean of previous pop MB (approx)
                Mdec_MB_old = mean(PopDec(:, UMB), 1);
                Mstep_MB = (topMB - Mdec_MB_old);  % [mu x D_MB]
                % weighted mean step
                Mstep_w = w * Mstep_MB;
                % update Mdec_MB and other CMA stats
                Mdec_MB = Mdec_MB_old + Mstep_w;
                Mdec(UMB) = Mdec_MB;  % sync global mean for bookkeeping

                % step-size evolution path (use chol of C_MB)
                C_MB = (C_MB + C_MB')/2 + 1e-12*eye(D_MB);
                U = chol(C_MB)';
                ps_MB = (1 - cs) * ps_MB + sqrt(cs*(2-cs)*mu_eff) * Mstep_w / U;
                sigma_MB = sigma_MB * exp(cs/ds * (norm(ps_MB)/ENN - 1))^0.3;

                hs = norm(ps_MB) / sqrt(1 - (1-cs)^(2*(ceil(Problem.FE/Problem.N)+1))) < hth;
                delta = (1 - hs) * cc * (2 - cc);
                pc_MB = (1 - cc) * pc_MB + hs * sqrt(cc*(2 - cc) * mu_eff) * Mstep_w;
                C_MB = (1 - c1 - cmu) * C_MB + c1 * (pc_MB' * pc_MB + delta * C_MB);
                for j = 1:mu
                    C_MB = C_MB + cmu * w(j) * (Mstep_MB(j,:)' * Mstep_MB(j,:));
                end
                C_MB = (C_MB + C_MB')/2; % symmetrize

                % Log MB indices
                deepc{gen} = UMB;

                % Optionally save deepc at the end
                if Problem.FE >= Problem.maxFE
                    runNo = 1;
                    file = fullfile(folder, sprintf('%s_%s_M%d_D%d_%s', class(Algorithm), class(Problem), Problem.M, Problem.D, 'MB'));
                    while exist([file, num2str(runNo), '.mat'], 'file') == 2
                        runNo = runNo + 1;
                    end
                    save([file, num2str(runNo), '.mat'], "deepc");
                end
            end % while
        end
    end
end
