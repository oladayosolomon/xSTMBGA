import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gymnasium as gym
import mo_gymnasium as mo_gym
import torch
from stable_baselines3 import A2C, PPO
#from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'




model_list={'A2C' : A2C , 'PPO' : PPO}
def evaluate_policy(model, env):
    env_lims={
         'Reacher-v4' : 50,
         'FrozenLake-v1' : 100,
         'HalfCheetah-v4' : 1000,
         'Hopper-v4' : 1000,
         'InvertedDoublePendulum-v4' : 1000,
         'LunarLander-v2' : 1000, #random
         'MountainCarContinuous-v0' : 1000,#random
    } 
    """Return mean fitness (sum of episodic rewards) for given model"""
    i_default=-1*env_lims[env.spec.id] if env.spec.id in env_lims else 1
    episode_rewards = []
    for _ in range(2):
        reward_sum = 0
        done = False
        obs,_ = env.reset()
        i=i_default
        while not done and i!=0:
            try :
                action,_state = model.predict(obs.copy())
            except AttributeError as e :
                action,_state = model.predict(obs)
            try :
                next_obs, vector_reward, done, ops, info = env.step(action)
            except TypeError as e:
                next_obs, vector_reward, done, ops, info = env.step(int(action))
            arr = np.array(vector_reward).ravel()
            reward_sum += arr
            obs=next_obs
            i+=1
            
            #print(reward_sum)
        episode_rewards.append(reward_sum)
        #print(type(episode_rewards))
    return (np.mean(episode_rewards,axis=0))
    
def decode(mean_params):
    #print(mean_params)
    #print(len(mean_params))

    labels = []
    lists = []
    for i in range(len(mean_params)):
        labels.append(list(mean_params.items())[i][0])
        lists.append(list(mean_params.items())[i][1])

    arrays = []
    array_shape = []
    for i in range(len(labels)):
        arrays.append(lists[i].cpu().detach().numpy().tolist())
        array_shape.append(list(lists[i].cpu().detach().numpy().shape))

    def flatten(list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
        return list_of_lists[:1] + flatten(list_of_lists[1:])

#   print('\n***************\n')
    comput_arrays = flatten(arrays)
    return comput_arrays, array_shape,labels


def encode(array,shapes,labels):
    k = 0
    new_dict = {}
    for i in range(len(shapes)):
        mult = 1
        for j in range(len(shapes[i])):
            mult *= shapes[i][j]
        temp = array[k:k+mult]
        temp = np.reshape(temp,tuple(shapes[i]))
        label = f'{labels[i]}'
        listt = torch.FloatTensor(temp)
        new_dict[label] = listt
        k+=mult
    return(new_dict)

def evaluate_env(env : str, agent : str, policy : str, weights: list) :
    try:
        
        if '_small' in env:
            env = env.replace('_small','')
            env=gym.make(env)
            weights=np.array(weights)
            model=model_list[agent](env=env,policy=policy,policy_kwargs={'net_arch':[32, 32]})
        else :
            env=gym.make(env)
            weights=np.array(weights)
            model=model_list[agent](env=env,policy=policy)
        mean_params = dict((key, value) for key, value in model.policy.state_dict().items() if ("policy" in key or "shared_net" in key or "action" in key))
        arr, shapes, labels=decode(mean_params)
        if len(weights.shape)==1 :
            candidate = encode(weights,shapes,labels)    
            model.policy.load_state_dict(candidate, strict=False)
            with open('yeahmen.txt','w') as fil:
                print(f'gota',file=fil)
            fitness = evaluate_policy(model, env)
            with open('yeahmen.txt','w') as fil:
                print(f'goto',file=fil)
            return -np.array(fitness)
        else :
            fitnesses=[]
            for j in range(len(weights)) :
                candidate = encode(weights[j],shapes,labels)
                model.policy.load_state_dict(candidate, strict=False)
                fitness = list(evaluate_policy(model,env))
                fitnesses.append(fitness)
            return -np.array(fitnesses)
    except Exception as e:
        print("this happened {e}")
        raise e

fitnesses=evaluate_env(env, agent, policy, weights)
