classdef FrozenLake < PROBLEM
% <single> <real> <large/none>
% DRL FrozenLake Task

%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem

        function Setting(obj)
            obj.M = 1;
            if isempty(obj.D); obj.D = 1732; end
            obj.lower     = ones(1,obj.D)*-1;
            obj.upper     = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);

        end
        %% Calculate objective values
        function PopObj = CalObj(obj,X)
            
            PopObj = pyrunfile("mat_eval_env.py","fitnesses",env="FrozenLake-v1_small", agent='A2C', policy='MlpPolicy', weights=X);
            PopObj = double(PopObj);
        end
        function Population = Evaluation(obj,varargin)
        %Evaluation - Evaluate multiple solutions.
        %
        %   P = obj.Evaluation(Dec) returns the SOLUTION objects based on
        %   the decision variables Dec. The objective values and constraint
        %   violations of the solutions are calculated automatically, and
        %   obj.FE is increased accordingly.
        %
        %   P = obj.Evaluation(Dec,Add) also sets the additional properties
        %   (e.g., velocity) of solutions.
        %
        %   This function is usually called after generating new solutions.
        %
        %   Example:
        %       Population = Problem.Evaluation(PopDec)
        %       Population = Problem.Evaluation(PopDec,PopVel)
            
            PopDec     = obj.CalDec(varargin{1});
            PopObj     = obj.CalObj(PopDec);
            PopCon     = obj.CalCon(PopDec);
            Population = SOLUTION(PopDec,PopObj,PopCon);
            obj.FE     = obj.FE + length(Population);
        end
    end
end 

