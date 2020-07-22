from typing import List, Tuple, Type, Union, Callable, Optional, Dict, Any
import torch as th
import torch.nn.functional as F
import numpy as np

from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.awac.policies import AWACPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm,trange


class AWAC(OffPolicyAlgorithm):
	"""
	Soft Actor-Critic (SAC)
	Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
	This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
	from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
	(https://github.com/rail-berkeley/softlearning/)
	and from Stable Baselines (https://github.com/hill-a/stable-baselines)
	Paper: https://arxiv.org/abs/1801.01290
	Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

	Note: we use double q target and not value target as discussed
	in https://github.com/hill-a/stable-baselines/issues/270

	:param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
	:param env: (GymEnv or str) The environment to learn from (if registered in Gym, can be str)
	:param learning_rate: (float or callable) learning rate for adam optimizer,
		the same learning rate will be used for all networks (Q-Values, Actor and Value function)
		it can be a function of the current progress remaining (from 1 to 0)
	:param buffer_size: (int) size of the replay buffer
	:param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
	:param batch_size: (int) Minibatch size for each gradient update
	:param tau: (float) the soft update coefficient ("Polyak update", between 0 and 1)
	:param gamma: (float) the discount factor
	:param train_freq: (int) Update the model every ``train_freq`` steps.
	:param gradient_steps: (int) How many gradient update after each step
	:param n_episodes_rollout: (int) Update the model every ``n_episodes_rollout`` episodes.
		Note that this cannot be used at the same time as ``train_freq``
	:param action_noise: (ActionNoise) the action noise type (None by default), this can help
		for hard exploration problem. Cf common.noise for the different action noise type.
	:param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
		inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
		Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
	:param target_update_interval: (int) update the target network every ``target_network_update_freq`` steps.
	:param target_entropy: (str or float) target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
	:param create_eval_env: (bool) Whether to create a second environment that will be
		used for evaluating the agent periodically. (Only available when passing string for the environment)
	:param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
	:param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
	:param seed: (int) Seed for the pseudo random generators
	:param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
		Setting it to auto, the code will be run on the GPU if possible.
	:param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
	"""

	def __init__(self, policy: Union[str, Type[AWACPolicy]],
				 env: Union[GymEnv, str],
				 learning_rate: Union[float, Callable] = 3e-4,
				 buffer_size: int = int(1e6),
				 learning_starts: int = 100,
				 batch_size: int = 256,
				 tau: float = 0.005,
				 gamma: float = 0.99,
				 train_freq: int = 1,
				 gradient_steps: int = 1,
				 n_episodes_rollout: int = -1,
				 action_noise: Optional[ActionNoise] = None,
				 ent_coef: Union[str, float] = 'auto',
				 target_update_interval: int = 1,
				 target_entropy: Union[str, float] = 'auto',
				 awr_use_mle_for_vf: bool = True,
				 beta: int = 50,
				 tensorboard_log: Optional[str] = None,
				 create_eval_env: bool = False,
				 policy_kwargs: Dict[str, Any] = None,
				 verbose: int = 0,
				 seed: Optional[int] = None,
				 device: Union[th.device, str] = 'auto',
				 _init_setup_model: bool = True):

		super().__init__(policy, env, AWACPolicy, learning_rate,
								  buffer_size, learning_starts, batch_size,
								  policy_kwargs, tensorboard_log, verbose, device,
								  create_eval_env=create_eval_env, seed=seed,
								  use_sde=False, sde_sample_freq=-1,
								  use_sde_at_warmup=False)

		self.target_entropy = target_entropy
		self.log_ent_coef = None  # type: Optional[th.Tensor]
		self.target_update_interval = target_update_interval
		self.tau = tau
		# Entropy coefficient / Entropy temperature
		# Inverse of the reward scale
		self.ent_coef = ent_coef
		self.target_update_interval = target_update_interval
		self.train_freq = train_freq
		self.gradient_steps = gradient_steps
		self.n_episodes_rollout = n_episodes_rollout
		self.action_noise = action_noise
		self.gamma = gamma
		self.ent_coef_optimizer = None
		self.awr_use_mle_for_vf = awr_use_mle_for_vf
		self.beta = beta
		self.bc_buffer = None

		if _init_setup_model:
			self._setup_model()

	def _setup_model(self) -> None:
		super()._setup_model()
		self._create_aliases()

		# Target entropy is used when learning the entropy coefficient
		if self.target_entropy == 'auto':
			# automatically set target entropy if needed
			self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
		else:
			# Force conversion
			# this will also throw an error for unexpected string
			self.target_entropy = float(self.target_entropy)

		# The entropy coefficient or entropy can be learned automatically
		# see Automating Entropy Adjustment for Maximum Entropy RL section
		# of https://arxiv.org/abs/1812.05905
		if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
			# Default initial value of ent_coef when learned
			init_value = 1.0
			if '_' in self.ent_coef:
				init_value = float(self.ent_coef.split('_')[1])
				assert init_value > 0., "The initial value of ent_coef must be greater than 0"

			# Note: we optimize the log of the entropy coeff which is slightly different from the paper
			# as discussed in https://github.com/rail-berkeley/softlearning/issues/37
			self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
			self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
		else:
			# Force conversion to float
			# this will throw an error if a malformed string (different from 'auto')
			# is passed
			self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

		self.bc_buffer = ReplayBuffer(int(1e4), self.observation_space,
                                        self.action_space, self.device)

	def _create_aliases(self) -> None:
		self.actor = self.policy.actor
		self.critic = self.policy.critic
		self.critic_target = self.policy.critic_target

	def pretrain_bc(self, gradient_steps: int, batch_size: int = 64):
		statistics = []
		with trange(gradient_steps) as t:
			for gradient_step in t:
				replay_data = self.bc_buffer.sample(batch_size, env=self._vec_normalize_env)
				dist = self.actor(replay_data.observations)
				actions_pi, log_prob = dist.log_prob_and_rsample()
				actor_loss = -log_prob.mean()
				actor_mse_loss = F.mse_loss(actions_pi.detach(),replay_data.actions)

				self.actor.optimizer.zero_grad()
				actor_loss.backward()
				self.actor.optimizer.step()

				statistics.append((actor_loss.item(),actor_mse_loss.item()))
				t.set_postfix(mse_loss=actor_mse_loss.item(),policy_loss=actor_loss.item())
		actor_losses,mse_losses = tuple(zip(*statistics))

		logger.record("pretrain/n_updates", self._n_updates, exclude='tensorboard')
		logger.record("pretrain/actor_loss", np.mean(actor_losses))
		logger.record("pretrain/actor_mse_loss", np.mean(mse_losses))

	def pretrain_rl(self, gradient_steps: int, batch_size: int = 64) -> None:
		statistics = []
		with trange(gradient_steps) as t:
			for gradient_step in t:
				replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
				stats = self.train_batch(replay_data)
				statistics.append(stats)
				self._n_updates += 1
				t.set_postfix(qf_loss=stats[1],policy_loss=stats[0])
		actor_losses,critic_losses,ent_coef_losses,ent_coefs = tuple(zip(*statistics))

		logger.record("pretrain/n_updates", self._n_updates, exclude='tensorboard')
		logger.record("pretrain/ent_coef", np.mean(ent_coefs))
		logger.record("pretrain/actor_loss", np.mean(actor_losses))
		logger.record("pretrain/critic_loss", np.mean(critic_losses))
		logger.record("pretrain/ent_coef_loss", np.mean(ent_coef_losses))

	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		statistics = []
		for gradient_step in range(gradient_steps):
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
			stats = self.train_batch(replay_data)
			statistics.append(stats)
			self._n_updates += 1
		actor_losses,critic_losses,ent_coef_losses,ent_coefs = tuple(zip(*statistics))

		logger.record("train/n_updates", self._n_updates, exclude='tensorboard')
		logger.record("train/ent_coef", np.mean(ent_coefs))
		logger.record("train/actor_loss", np.mean(actor_losses))
		logger.record("train/critic_loss", np.mean(critic_losses))
		logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

	def train_batch(self,replay_data):
		# Action by the current actor for the sampled state
		dist = self.actor(replay_data.observations)
		actions_pi, log_prob = dist.log_prob_and_rsample()
		actor_mle = dist.mean

		"""ent_coeff loss"""
		ent_coef_loss = None
		if self.ent_coef_optimizer is not None:
			# Important: detach the variable from the graph
			# so we don't change it with other losses
			# see https://github.com/rail-berkeley/softlearning/issues/60
			ent_coef = th.exp(self.log_ent_coef.detach())
			ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
		else:
			ent_coef = self.ent_coef_tensor

		"""q loss"""
		with th.no_grad():
			# Select action according to policy
			next_dist = self.actor(replay_data.next_observations)
			next_actions, next_log_prob = next_dist.log_prob_and_rsample()
			# Compute the target Q value
			target_q1, target_q2 = self.critic_target(replay_data.next_observations, next_actions)
			target_q = th.min(target_q1, target_q2)
			target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q
			# td error + entropy term
			# q_backup = target_q - ent_coef * next_log_prob
			q_backup = target_q
		# Get current Q estimates
		# using action from the replay buffer
		current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)
		# Compute critic loss
		critic_loss = 0.5 * (F.mse_loss(current_q1, q_backup) + F.mse_loss(current_q2, q_backup))

		"""action loss"""
		# Advantage-weighted regression
		# if self.awr_use_mle_for_vf:
		# 	v1_pi,v2_pi = self.critic(replay_data.observations, actor_mle)
		# 	v_pi = th.min(v1_pi, v2_pi)
		# else:
		# 	v1_pi,v2_pi = self.critic(replay_data.observations, actions_pi)
		# 	v_pi = th.min(v1_pi, v2_pi)
		q_adv = th.min(current_q1,current_q2)
		v1_pi,v2_pi = self.critic(replay_data.observations, actor_mle)
		v_pi = th.min(v1_pi, v2_pi)
		# q_adv = th.min(*self.critic(replay_data.observations, actions_pi))
		score = q_adv - v_pi
		weights = F.softmax(score/self.beta,dim=0)
		# actor_loss = ent_coef * log_prob.mean()
		actor_logpp = dist.log_prob(replay_data.actions)
		actor_loss = (-actor_logpp * len(weights)*weights.detach()).mean()

		"""Updates"""
		# Optimize entropy coefficient, also called
		# entropy temperature or alpha in the paper
		if ent_coef_loss is not None:
			self.ent_coef_optimizer.zero_grad()
			ent_coef_loss.backward()
			self.ent_coef_optimizer.step()
		# Optimize the critic
		self.critic.optimizer.zero_grad()
		critic_loss.backward()
		self.critic.optimizer.step()
		# Optimize the actor
		self.actor.optimizer.zero_grad()
		actor_loss.backward()
		self.actor.optimizer.step()

		# Update target networks
		if self._n_updates % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		if ent_coef_loss is None:
			ent_coef_loss = th.tensor([0])
		return actor_loss.item(),critic_loss.item(),ent_coef_loss.item(),ent_coef.item()

	def learn(self,
			  total_timesteps: int,
			  callback: MaybeCallback = None,
			  log_interval: int = 4,
			  eval_env: Optional[GymEnv] = None,
			  eval_freq: int = -1,
			  n_eval_episodes: int = 5,
			  tb_log_name: str = "AWAC",
			  eval_log_path: Optional[str] = None,
			  reset_num_timesteps: bool = True) -> OffPolicyAlgorithm:

		total_timesteps, callback = self._setup_learn(total_timesteps, eval_env, callback, eval_freq,
													  n_eval_episodes, eval_log_path, reset_num_timesteps,
													  tb_log_name)
		callback.on_training_start(locals(), globals())

		self.pretrain_bc(int(1e3),batch_size=self.batch_size)
		observations,actions,next_observations,rewards,dones = self.bc_buffer.observations,self.bc_buffer.actions,self.bc_buffer.next_observations,self.bc_buffer.rewards,self.bc_buffer.dones
		for data in zip(observations,next_observations,actions,rewards,dones):
			self.replay_buffer.add(*data)
		self.pretrain_rl(int(1e4),batch_size=self.batch_size)

		while self.num_timesteps < total_timesteps:
			rollout = self.collect_rollouts(self.env, n_episodes=self.n_episodes_rollout,
											n_steps=self.train_freq, action_noise=self.action_noise,
											callback=callback,
											learning_starts=self.learning_starts,
											replay_buffer=self.replay_buffer,
											log_interval=log_interval)

			if rollout.continue_training is False:
				break

			self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

			if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
				gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
				self.train(gradient_steps, batch_size=self.batch_size)

		callback.on_training_end()
		return self

	def excluded_save_params(self) -> List[str]:
		"""
		Returns the names of the parameters that should be excluded by default
		when saving the model.

		:return: (List[str]) List of parameters that should be excluded from save
		"""
		# Exclude aliases
		return super().excluded_save_params() + ["actor", "critic", "critic_target"]

	def get_torch_variables(self) -> Tuple[List[str], List[str]]:
		"""
		cf base class
		"""
		state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
		saved_tensors = ['log_ent_coef']
		if self.ent_coef_optimizer is not None:
			state_dicts.append('ent_coef_optimizer')
		else:
			saved_tensors.append('ent_coef_tensor')
		return state_dicts, saved_tensors
