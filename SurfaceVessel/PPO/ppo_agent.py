import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from ppo_memory import PPOMemory
from ppo_networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/'):
        self.gamma = gamma
        self.input_dims = input_dims
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)
        self.log_std = tf.Variable(
            initial_value=-0.5 * np.ones(n_actions, dtype=np.float32),
            trainable=True
        )

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        #For PPO training you dont need full keras serialisation
        #self.actor.save(self.chkpt_dir + 'actor.keras')   
        #self.critic.save(self.chkpt_dir + 'critic.keras')
        self.actor.save_weights(self.chkpt_dir + 'actor.weights.h5')
        self.critic.save_weights(self.chkpt_dir + 'critic.weights.h5')
        

    def load_models(self):
        print('... loading models ...')
        #self.actor = keras.models.load_model(self.chkpt_dir + 'actor.keras')  #keras 3 requires the .keras extension
        #self.critic = keras.models.load_model(self.chkpt_dir + 'critic.keras')
        # Build networks first
        dummy_state = np.zeros(
            (1, self.input_dims[0]),
            dtype=np.float32
        )

        self.actor(dummy_state)
        self.critic(dummy_state)

        self.actor.load_weights(self.chkpt_dir + 'actor.weights.h5')
        self.critic.load_weights(self.chkpt_dir + 'critic.weights.h5')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        mu = self.actor(state)  #Mean for the distro
        sigma = tf.exp(self.log_std)
        dist = tfp.distributions.Normal(mu, sigma)   #Gaussian Distribution for Probability density over infinite space

        #probs = self.actor(state)
        #dist = tfp.distributions.Categorical(probs)  #This yields discrete action space

        action = dist.sample()
        #action += 0.3
        action = tf.clip_by_value(action, -1, 1)  #Clipping it cuz the env expects from -1 to 1
        #log_prob = dist.log_prob(action)  #Action has two dimensions, so need to apply log_prob for both the dimensions
        log_prob = tf.reduce_sum(
            dist.log_prob(action),
            axis=-1  #This tells to sum up the last dimensions of the matrix
        )
        value = self.critic(state)
        value = tf.squeeze(value)
        value = float(value)

        action = action.numpy()[0]
        #value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
                
            advantage = (
                    advantage - np.mean(advantage)
                ) / (np.std(advantage) + 1e-8)

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    #probs = self.actor(states)
                    #dist = tfp.distributions.Categorical(probs)
                    #We need a Gaussian Distro
                    mu = self.actor(states)
                    sigma = tf.exp(self.log_std)
                    dist = tfp.distributions.Normal(mu, sigma)
                    new_probs = tf.reduce_sum(dist.log_prob(actions),axis = -1)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    #critic_loss = keras.losses.MSE(critic_value, returns)
                    critic_loss = tf.reduce_mean(
                        keras.losses.MSE(
                            critic_value,
                            returns
                        )
                    )

                actor_params = self.actor.trainable_variables + [self.log_std]
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))

        self.memory.clear_memory()
