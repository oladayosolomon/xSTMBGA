function UMB= LSMB(Population,Problem)

% 'dis' represents discrete data, 'con' denotes continues data 
data_type='con';     

% Name of algorithm
alg_name='STMB';     %IAMB  %GS   %interIAMB %FBED


 %alg_name='GS'; alg_name='IAMB'; alg_name='interIAMB'; alg_name='IAMBnPC';
 %alg_name='interIAMBnPC'; alg_name='FastIAMB'; alg_name='FBED'; alg_name='EAMB'; alg_name='MMMB'; alg_name='HITONMB';
 %alg_name='PCMB'; alg_name='IPCMB'; alg_name='MBOR'; alg_name='STMB'; alg_name='BAMB'; alg_name='EEMB';  alg_name='MBFS'; alg_name='CFS_MI';       


% Significance level
alpha=0.05;

% Index of target node. If it is global structure learning, this parameter is not needed
target = Problem.D + 1;   

% Load data according to the path
% data needs to start from 0
tdata = Population.objs;

for i = 1: Problem.M

data = [Population.decs,tdata(:,i)];
% Causal_Learner
[Result{i},~,~] = Causal_Learner(alg_name,data,data_type,alpha,target);
end
temp = cell2mat(Result);
UMB = unique(temp);

end

% Markov blanket learning 
% Result1 is learned target's Markov blanket.
% Result2 is the number of conditional independence tests
% Result3 is running time

















