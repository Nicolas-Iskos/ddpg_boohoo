import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

torch.manual_seed(0)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def mlp2(sizes, activation=nn.Tanh, output_activation=nn.Tanh):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class mlp3(nn.Module):
    # Build a feedforward neural network.
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        if(x.dim() == 1):
            return torch.zeros((1))
        else:
            return torch.zeros((x.shape[0],1))

def train(env_name='CartPole-v0', hidden_sizes=[32], critic_lr=0, actor_lr=0, 
          epochs=100000, batch_size=10000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    act_dim = 1 # cuz you either go left or right
    act_output_dim = 1

    # make core of policy network
    actor = mlp2(sizes=[obs_dim]+hidden_sizes+[act_output_dim])
    #actor = mlp3()

    # 1 is the reward dim
    critic = mlp(sizes=[obs_dim+act_output_dim]+hidden_sizes+[1])
    
    critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
    actor_optimizer = Adam(actor.parameters(), lr=actor_lr)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(act):
        return int(act > 0)

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_critic_loss(obs, obsp, act, weights):
        pre_pro_actp = actor(obsp)
        actp = pre_pro_actp
        y = weights.reshape(-1, 1) + critic(torch.concat((obsp, actp), dim=1))
        y2 = critic(torch.concat((obsp, actp), dim=1))
        y1 = critic(torch.concat((obs, act), dim=1)) 
        #print("prediction=", y1.shape, obs.shape, act.shape)
        #print(obs[0:50,0])
        #print(obsp[0:50,0])
        print(y1[0:50,0])
        #print(y[0:50,0])
        print(y2[0:50,0])
        #print(torch.concat((obs,act),dim=1))
        #print(torch.concat((obsp,actp),dim=1))
        #print(act)
        l = (y1- y)**2
        return l.mean()

    def compute_actor_loss(obs):
        pre_pro_act = actor(obs)
        return critic(torch.concat((obs, pre_pro_act), dim=1)).mean()

    # for training policy
    def train_one_epoch(epoch_idx):
        # for now, the entire run is sampled for gradient calculation (B = N)
        batch_obs = []          # for observations
        batch_obsp = []         # for the other part of the transitions
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        #obs = env.reset()       # first obs comes from starting distribution
        obs = env.reset()[0]       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = actor(torch.as_tensor(obs, dtype=torch.float32))

            proc_act = get_action(act)
   
            obs, rew, done, _, _ = env.step(proc_act)

            batch_obsp.append(obs.copy())
            batch_acts.append(act.item())
            ep_rews.append(rew)
    
            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        critic_optimizer.zero_grad()
        critic_batch_loss = compute_critic_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                                obsp=torch.as_tensor(batch_obsp, dtype=torch.float32),
                                                act=torch.as_tensor(batch_acts, dtype=torch.float32).reshape((-1, act_output_dim)),
                                                weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        critic_batch_loss.backward()
        critic_optimizer.step()
        
        actor_optimizer.zero_grad()
        actor_batch_loss = compute_actor_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32))
        if(epoch_idx > 50 and epoch_idx % 100 == 0):
            actor_batch_loss.backward()
            actor_optimizer.step()
        

        return critic_batch_loss, actor_batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        critic_batch_loss, actor_batch_loss, batch_rets, batch_lens = train_one_epoch(i)
        print('epoch: %3d \t critic loss: %.3f \t actor loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, critic_batch_loss, actor_batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--critic_lr', type=float, default=1e-2)
    parser.add_argument('--actor_lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, critic_lr=args.critic_lr, actor_lr=args.actor_lr)