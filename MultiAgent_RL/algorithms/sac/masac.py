# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from algorithms.sac.sac import SACAgent
import torch
from utilities.utilities import soft_update, transpose_to_tensor, transpose_list, gumbel_softmax
import numpy as np

'''
An addaption from:

Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

'''

class MASAC:
    def __init__(self, num_agents = 3, num_landmarks = 1, landmark_depth=15., discount_factor=0.95, tau=0.02, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn = True, alpha = 0.2, automatic_entropy_tuning = True, dim_1=64, dim_2=32):
        super(MASAC, self).__init__()
        
        # ([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [entity_range] + [entity_depth] + [agent.state.p_pos_origin]) + action(for critic not actor)
        #in_actor = 1*2*2 + num_landmarks*2 + (num_agents-1)*2 + num_landmarks + 1*num_landmarks + 2 +1#test with target depth and agent's origin for science
        obs_dim = 5 + (num_agents - 1) * 4

        in_actor = obs_dim

        hidden_in_actor = dim_2
        hidden_out_actor = int(hidden_in_actor/2)
        out_actor = 2 #each agent have 2 continuous actions on x-y plane
        #in_critic = in_actor * num_agents # the critic input is all agents concatenated
        action_dim = 2

        in_critic = (obs_dim + action_dim) * num_agents
        hidden_in_critic = dim_2
        hidden_out_critic = int(hidden_in_critic/2)
        #RNN
        rnn_num_layers = 2 #two stacked RNN to improve the performance (default = 1)
        rnn_hidden_size_actor = dim_1
        rnn_hidden_size_critic = dim_1
        
        # print('Actor NN configuration:')
        # print('Input nodes number:            ',in_actor)
        # print('Hidden 1st layer nodes number: ',hidden_in_actor)
        # print('Hidden 2nd layer nodes number: ',hidden_out_actor)
        # print('Output nodes number:           ',out_actor)
        # print('RNN hidden size actor :        ',rnn_hidden_size_actor)
        # print('Critic NN configuration:')
        # print('Input nodes number:            ',in_critic)
        # print('Hidden 1st layer nodes number: ',hidden_in_critic)
        # print('Hidden 2nd layer nodes number: ',hidden_out_critic)
        # print('Output nodes number:           ',out_actor)
        # print('RNN hidden size critic:        ',rnn_hidden_size_critic)
        
        self.masac_agent = [SACAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device, rnn = rnn, alpha = alpha, automatic_entropy_tuning=automatic_entropy_tuning) for _ in range(num_agents)]
        # self.masac_agent = [DDPGAgent(14, 128, 128, 2, 48, 128, 128, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device) for _ in range(num_agents)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.iter_delay = 0
        
        self.policy_freq = 2
        self.num_agents = num_agents
        
        #initial priority for the experienced replay buffer
        self.priority = 1.
        
        #device 'cuda' or 'cpu'
        self.device = device
        
        #To update alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [sac_agent.actor for sac_agent in self.masac_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [sac_agent.target_actor for sac_agent in self.masac_agent]
        return target_actors

    def act(self, his_all_agents, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions_next = [agent.act(his, obs, noise) for agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents)]
        return actions_next

    def act_prob(self, his_all_agents, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        actions_next = []
        log_probs = []
        for sac_agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents):
            action, log_prob = sac_agent.act_prob(his, obs)
            log_prob = log_prob.view(-1)
            actions_next.append(action)
            log_probs.append(log_prob)
        # target_actions_next = [sac_agent.target_actor.sample_normal(his, obs, noise) for sac_agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents)]
        # for i,aux in enumerate(log_probs):
        #     log_probs[i]=aux.view(-1,1)
        return actions_next, log_probs

    def update(self, samples, agent_number, logger):

        obs, action, reward, next_obs, done = samples
        obs = [
            torch.FloatTensor(np.array(o)).to(self.device)
            for o in obs
        ]

        action = [
            torch.FloatTensor(np.array(a)).to(self.device)
            for a in action
        ]

        reward = [
            torch.FloatTensor(np.array(r)).to(self.device)
            for r in reward
        ]

        next_obs = [
            torch.FloatTensor(np.array(no)).to(self.device)
            for no in next_obs
        ]

        done = [
            torch.FloatTensor(np.array(d)).to(self.device)
            for d in done
        ]

        # =====================================================
        # CONCATENATE GLOBAL STATE
        # =====================================================

        obs_full = torch.cat(obs, dim=1)
        obs_full = obs_full.to(self.device)

        next_obs_full = torch.cat(next_obs, dim=1)
        next_obs_full = next_obs_full.to(self.device)

        action_full = torch.cat(action, dim=1)
        action_full = action_full.to(self.device)

        obs_act_full = torch.cat(
            (obs_full, action_full),
            dim=1
        )

        agent = self.masac_agent[agent_number]

        agent.critic_optimizer.zero_grad()

        # =====================================================
        # NEXT ACTIONS
        # =====================================================

        actions_next = []

        log_probs = []

        for sac_agent, obs_i in zip(self.masac_agent, next_obs):

            action_i, log_prob_i = sac_agent.act_prob(
                None,
                obs_i
            )

            log_prob_i = log_prob_i.view(-1)

            actions_next.append(action_i)

            log_probs.append(log_prob_i)

        actions_next = torch.cat(
            actions_next,
            dim=1
        ).to(self.device)

        next_obs_act_full = torch.cat(
            (next_obs_full, actions_next),
            dim=1
        )

        # =====================================================
        # TARGET Q
        # =====================================================

        with torch.no_grad():

            target_Q1, target_Q2 = agent.target_critic(
                None,
                next_obs_act_full.to(self.device)
            )

            target_V = (
                torch.min(target_Q1, target_Q2)
                - agent.alpha *
                log_probs[agent_number].view(-1, 1)
            )

            target_Q = (
                reward[agent_number].view(-1, 1).to(self.device)
                + self.discount_factor
                * target_V
                * (
                    1
                    - done[agent_number]
                    .view(-1, 1)
                    .to(self.device)
                )
            )

        # =====================================================
        # CURRENT Q
        # =====================================================

        current_Q1, current_Q2 = agent.critic(
            None,
            obs_act_full.to(self.device)
        )

        # =====================================================
        # CRITIC LOSS
        # =====================================================

        loss_mse = torch.nn.MSELoss()

        critic_loss = (
            loss_mse(current_Q1, target_Q.detach())
            +
            loss_mse(current_Q2, target_Q.detach())
        )

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            agent.critic.parameters(),
            0.5
        )

        agent.critic_optimizer.step()

        # =====================================================
        # ACTOR UPDATE
        # =====================================================

        if self.iter_delay % self.policy_freq == 0:

            agent.actor_optimizer.zero_grad()

            actions, log_probs_actor = \
                self.masac_agent[agent_number] \
                .actor.sample_normal(
                    None,
                    obs[agent_number].to(self.device)
                )

            log_probs_actor = log_probs_actor.view(-1)

            q_actions = []

            q_log_probs = []

            for i, ob in enumerate(obs):

                if i == agent_number:

                    q_actions.append(actions)

                    q_log_probs.append(log_probs_actor)

                else:

                    actions_aux, log_probs_aux = \
                        self.masac_agent[i] \
                        .actor.sample_normal(
                            None,
                            ob.to(self.device)
                        )

                    log_probs_aux = log_probs_aux.view(-1)

                    q_actions.append(actions_aux.detach())

                    q_log_probs.append(
                        log_probs_aux.detach()
                    )

            q_actions = torch.cat(q_actions, dim=1)

            obs_q_full = torch.cat(
                (
                    obs_full.to(self.device),
                    q_actions
                ),
                dim=1
            )

            actor_Q1, actor_Q2 = agent.critic(
                None,
                obs_q_full
            )

            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (
                agent.alpha
                * q_log_probs[agent_number].view(-1, 1)
                - actor_Q
            ).mean()

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                agent.actor.parameters(),
                0.5
            )

            agent.actor_optimizer.step()

            # =================================================
            # ALPHA UPDATE
            # =================================================

            if self.automatic_entropy_tuning:

                alpha_loss = -(
                    agent.log_alpha
                    * (
                        q_log_probs[agent_number]
                        .view(-1, 1)
                        + agent.target_entropy
                    ).detach()
                ).mean()

                agent.alpha_optimizer.zero_grad()

                alpha_loss.backward()

                agent.alpha_optimizer.step()

                agent.alpha = agent.log_alpha.exp()

            # =================================================
            # LOGGING
            # =================================================

            al = actor_loss.cpu().detach().item()

            cl = critic_loss.cpu().detach().item()

            if logger is not None:

                logger.add_scalars(
                    'agent%i/losses' % agent_number,
                    {
                        'critic loss': cl,
                        'actor_loss': al
                    },
                    self.iter
                )

    def update_targets(self):
        """soft update targets"""
        self.iter += 1 #this doesnt work as well as the other test 80
        self.iter_delay += 1
        # ----------------------- update target networks ----------------------- #
        for sac_agent in self.masac_agent:
            # soft_update(sac_agent.target_actor, sac_agent.actor, self.tau)
            soft_update(sac_agent.target_critic, sac_agent.critic, self.tau)
            
            
            




