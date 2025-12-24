classdef InvertedDoublePendulum < PROBLEM
% <single> <real> <large/none>
% DRL InvertedDoublePendulum Task
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem

        function Setting(obj)
            obj.M = 1;
            if isempty(obj.D); obj.D = 1473; end
            obj.lower     = ones(1,obj.D)*-1;
            obj.upper     = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);

        end
        %% Calculate objective values
        function PopObj = CalObj(obj,X)
            
            PopObj = pyrunfile("mat_eval_env.py","fitnesses",env="InvertedDoublePendulum-v4_small", agent='A2C', policy='MlpPolicy', weights=X);
            PopObj = double(PopObj);
        end
        function Population = Evaluation(obj,varargin)
              
            PopDec     = obj.CalDec(varargin{1});
            PopObj     = obj.CalObj(PopDec);
            PopCon     = obj.CalCon(PopDec);
            Population = SOLUTION(PopDec,PopObj,PopCon);
            obj.FE     = obj.FE + length(Population);
        end
    end
end 

