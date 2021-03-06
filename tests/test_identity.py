import numpy as np
import pytest

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.identity_env import (IdentityEnvBox, IdentityEnv,
                                                   IdentityEnvMultiBinary, IdentityEnvMultiDiscrete)

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise


DIM = 4


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env", [IdentityEnv(DIM), IdentityEnvMultiDiscrete(DIM), IdentityEnvMultiBinary(DIM)])
def test_discrete(model_class, env):
    env = DummyVecEnv([lambda: env])
    model = model_class('MlpPolicy', env, gamma=0.5, seed=1).learn(3000)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)
    obs = env.reset()

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3])
def test_continuous(model_class):
    env = IdentityEnvBox(eps=0.5)

    n_steps = {
        A2C: 3500,
        PPO: 3000,
        SAC: 700,
        TD3: 500
    }[model_class]

    kwargs = dict(
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=0,
        gamma=0.95
    )
    if model_class in [TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        kwargs['action_noise'] = action_noise

    model = model_class('MlpPolicy', env, **kwargs).learn(n_steps)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)
