#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import numpy as np
import apex

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            #tmp_state = config.state_normalizer(np.stack([self._state])) 
            #if config.half:
            #    tmp_state.astype(np.float16)
            #q_values = self._network(tmp_state)
            q_values = self._network(config.state_normalizer(np.stack([self._state])), half=config.half)
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, int(done), info]
        self._total_steps += 1
        if done:
            next_state = self._task.reset()
        self._state = next_state
        return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)
        self.network = config.network_fn()
        if config.half:
            self.network = self.network.half()
        self.network.share_memory()
        self.target_network = config.network_fn()
        if config.half:
            self.target_network = self.target_network.half()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        if config.half:
            self.optimizer = apex.fp16_utils.FP16_Optimizer(self.optimizer, static_loss_scale=64.0)

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        q = self.network(state, half=self.config.half)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if (self.total_steps > self.config.exploration_steps):
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network(next_states, half=self.config.half).detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states, half=self.config.half), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            if self.config.half:
                terminals = terminals.half()
                rewards = rewards.half()
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            save_gradient = False
            if (self.config.gradient_step and (self.total_steps % self.config.gradient_step == 0)):
                save_gradient = True
            q = self.network(states, half=self.config.half, save_gradient=save_gradient)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.writer.add_scalar("Train_mean_loss", loss, self.total_steps)

            #config.logger.info("Training mean loss: %d" % (loss.clone().cpu().data)) 
            #if np.isnan(loss.item()):
            #    raise ValueError("NaN in Loss!!!!!!!!!!!!!!!!!!!!!")
            self.optimizer.zero_grad()
            if config.half:
                self.optimizer.backward(loss)
            else:
                loss.backward()
            if (self.config.gradient_step and (self.total_steps % self.config.gradient_step == 0)):
                for name, para in self.network.named_parameters():
                    log2_grad = []
                    for item in para.grad.clone().cpu().data.numpy().flatten(-1):
                        if np.isnan(item):
                            raise ValueError("whoops, NaN")
                        if item == 0:
                            log2_grad.append(0)
                        else:
                            log2_grad.append(int(np.log2(abs(item))))
                    self.writer.add_histogram(name + "before_clip", np.array(log2_grad), self.total_steps)
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            #tmp_grads = []
            if (self.config.gradient_step and (self.total_steps % self.config.gradient_step == 0)):
                for name, para in self.network.named_parameters():
                    log2_grad = []
                    tmp_grad = []
                    for item in para.grad.clone().cpu().data.numpy().flatten(-1):
                        if np.isnan(item):
                            raise ValueError("whoooooooooooooooooo, NaN")
                        tmp_grad.append(item)
                        if item == 0:
                            log2_grad.append(0)
                            self.config.gradient_summary.append(0)
                        else:
                            log2_grad.append(int(np.log2(abs(item))))
                            self.config.gradient_summary.append(int(np.log2(abs(item))))
                    self.writer.add_histogram(name + "after_clip", np.array(log2_grad), self.total_steps)
                    print ("Name: %s, Mean gradient: %.3f" % (name, np.mean(np.array(tmp_grad))))
            if self.total_steps == 1:
                self.writer.add_histogram("normal gradient 1", np.array(self.config.gradient_summary), self.total_steps)
                self.writer.add_histogram("activation gradient 1", np.array(self.config.activation_gradient_summary), self.total_steps)
            if self.total_steps == int(1e7):
                self.writer.add_histogram("normal gradient 1e7", np.array(self.config.gradient_summary), self.total_steps)
                self.writer.add_histogram("activation gradient 1e7", np.array(self.config.activation_gradient_summary), self.total_steps)
            if self.total_steps == int(1e7 + 5e6):
                self.writer.add_histogram("normal gradient 1.5e7", np.array(self.config.gradient_summary), self.total_steps)
                self.writer.add_histogram("activation gradient 1.5e7", np.array(self.config.activation_gradient_summary), self.total_steps)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
