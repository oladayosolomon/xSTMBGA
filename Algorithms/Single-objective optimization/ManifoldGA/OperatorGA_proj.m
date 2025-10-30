function Offspring = OperatorGA_proj(Parents, parameters)
    [proC, disC, proM, disM] = deal(parameters{:});
    [N, D] = size(Parents);
    Offspring = zeros(N, D);

    %% Simulated Binary Crossover (SBX)
    for i = 1:2:N
        p1 = Parents(i,:);
        if i < N
            p2 = Parents(i+1,:);
        else
            p2 = Parents(randi(N),:);
        end
        beta = zeros(1,D);
        mu = rand(1,D);
        beta(mu <= 0.5) = (2*mu(mu <= 0.5)).^(1/(disC+1));
        beta(mu > 0.5)  = (2-2*mu(mu > 0.5)).^(-1/(disC+1));
        beta = beta .* ((-1).^randi([0,1],1,D));
        beta(rand(1,D) > proC) = 1;
        c1 = 0.5*((1+beta).*p1 + (1-beta).*p2);
        c2 = 0.5*((1-beta).*p1 + (1+beta).*p2);
        Offspring(i,:) = c1;
        if i < N
            Offspring(i+1,:) = c2;
        end
    end

    %% Polynomial Mutation
    for i = 1:N
        for j = 1:D
            if rand < proM/D
                delta = rand;
                if delta < 0.5
                    delta = (2*delta)^(1/(disM+1)) - 1;
                else
                    delta = 1 - (2*(1-delta))^(1/(disM+1));
                end
                Offspring(i,j) = Offspring(i,j) + delta;
            end
        end
    end
end
