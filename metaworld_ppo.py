import numpy as np
import gym
from gym import spaces
import mujoco_py
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.bench import Monitor
import os 
import logging
import warnings

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

class MetaWorld(gym.Env):
  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, reward_scale=1):
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    self._task = name.replace("_", "-") + "-v2-goal-observable"
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[self._task]()
    self._env = env
    self._action_repeat = action_repeat
    self._size = size
    self._camera = camera
    self._reward_scale = reward_scale
    self._camera_settings = {
      "lexa": dict(distance=0.6, lookat=[0, 0.65, 0], azimuth=90, elevation=41+180),
      "latco_hammer": dict(distance=0.8, lookat=[0.2, 0.65, -0.1], azimuth=220, elevation=-140),
      "latco_others": dict(distance=2.6, lookat=[1.1, 1.1, -0.1], azimuth=205, elevation=-165)
    }
    if self._camera == "lexa" or self._camera == "latco_hammer" or self._camera == "latco_others":
      self._cam = self._camera_settings[self._camera]
      self._env.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
    self.observation_space = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    self.action_space = self._env.action_space

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._action_repeat):
      _, reward, done, info = self._env.step(action)
      total_reward += reward 
      if self._env.curr_path_length == self._env.max_path_length: #https://github.com/rlworkgroup/metaworld/issues/236
        break
    if self._camera == "lexa" or self._camera == "latco_hammer" or self._camera == "latco_others":
      self._env.viewer.cam.distance, self._env.viewer.cam.azimuth, self._env.viewer.cam.elevation = self._cam["distance"], self._cam["azimuth"], self._cam["elevation"]
      self._env.viewer.cam.lookat[0], self._env.viewer.cam.lookat[1], self._env.viewer.cam.lookat[2] = self._cam["lookat"][0], self._cam["lookat"][1], self._cam["lookat"][2] 
      self._env.viewer.render(self._size[0], self._size[1])
      img = self._env.viewer.read_pixels(self._size[0], self._size[1])[0]
    else:      
      img = self._env.render(offscreen=True, resolution=self._size, camera_name=self._camera)
    done = self._env.curr_path_length == self._env.max_path_length
    return img, total_reward, done, info

  def reset(self):
    self._env.reset()
    if self._camera == "lexa" or self._camera == "latco_hammer" or self._camera == "latco_others":
      self._env.viewer.cam.distance, self._env.viewer.cam.azimuth, self._env.viewer.cam.elevation = self._cam["distance"], self._cam["azimuth"], self._cam["elevation"]
      self._env.viewer.cam.lookat[0], self._env.viewer.cam.lookat[1], self._env.viewer.cam.lookat[2] = self._cam["lookat"][0], self._cam["lookat"][1], self._cam["lookat"][2] 
      self._env.viewer.render(self._size[0], self._size[1])
      img = self._env.viewer.read_pixels(self._size[0], self._size[1])[0]
    else:      
      img = self._env.render(offscreen=True, resolution=self._size, camera_name=self._camera)
    return img

  def close(self):
    pass

log_dir = "~/logs/ppo/reach/"
os.makedirs(log_dir, exist_ok=True)    
env = MetaWorld(name='reach', action_repeat=2, camera='latco_others')
env = Monitor(env, log_dir)
print("Environment created!")
check_env(env, warn=True)
env = make_vec_env(lambda: env, n_envs=1)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model = PPO2('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e6), callback=callback)
model.save("ppo_reach_metaworld")
# Test the trained agent
obs = env.reset()
n_steps = 1000
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  obs, reward, done, info = env.step(action)
  print('reward=', reward, 'done=', done)
  #env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break