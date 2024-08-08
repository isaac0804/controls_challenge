import numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import BaseController
from trajectory_loader import TrajectoryDataLoader

class TinyPhysicsEnv:
    def __init__(self, data_dir, model_path):
        self.data_loader = TrajectoryDataLoader(data_dir)
        self.model_path = model_path
        self.tinyphysicsmodel = TinyPhysicsModel(self.model_path, debug=False)
        self.simulator = None
        self.step_idx = 0
        self.max_steps = 50 # Adjust as needed
        self.action_space = np.linspace(-2, 2, 21)  # 21 discrete actions from -2 to 2

    def reset(self):
        class DummyController(BaseController):
            def update(self, target_lat_accel, current_lat_accel, state, future_plan):
                return 0  # Dummy action, will be overwritten by the environment

        random_file = self.data_loader.get_random_file()
        # data = self.data_loader.load_single_file(random_file)
        # print(random_file)
        # print(data)
        self.simulator = TinyPhysicsSimulator(self.tinyphysicsmodel, random_file.as_posix(), controller=DummyController(), debug=False)
        self.step_idx = 0
        return self._get_state()

    def step(self, action_idx):
        action = self.action_space[action_idx]
        self.simulator.action_history.append(action)
        self.simulator.sim_step(self.step_idx)
        self.step_idx += 1

        next_state = self._get_state()
        reward = self._compute_reward()
        done = self.step_idx >= self.max_steps

        return next_state, reward, done, {}

    def _get_state(self):
        return np.array([
            self.simulator.state_history[-1].roll_lataccel,
            self.simulator.state_history[-1].v_ego,
            self.simulator.state_history[-1].a_ego,
            self.simulator.target_lataccel_history[-1],
            self.simulator.current_lataccel
        ])

    def _compute_reward(self):
        target = self.simulator.target_lataccel_history[-1]
        current = self.simulator.current_lataccel
        prev_current = self.simulator.current_lataccel_history[-2] if len(self.simulator.current_lataccel_history) > 1 else current

        lataccel_cost = -((target - current) ** 2)
        jerk_cost = -((current - prev_current) / 0.1) ** 2  # Assuming 10 Hz frequency (0.1 second interval)

        return lataccel_cost * 50 + jerk_cost  # Weighted sum of costs, adjust weights as needed

# Usage example:
# env = TinyPhysicsEnv('data/', 'models/tinyphysics.onnx')
# state = env.reset()
# for _ in range(400):
#     action = agent.select_action(state)  # Your MuZero agent
#     next_state, reward, done, _ = env.step(action)
#     if done:
#         break
#     state = next_state