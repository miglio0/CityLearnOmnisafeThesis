import argparse
import yaml
from abc import ABC


class Config(ABC):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Configurations for Constrained RL on CityLearn")
        self._add_args()

    def _add_args(self):
        # Seed
        self.parser.add_argument('--seed', type=int, default=1, help="Experiment seed")

        # CityLearn dataset
        self.parser.add_argument('--data', type=str, default='citylearn_challenge_2023_phase_1', help="CityLearn dataset")


class TrainConfig(Config):
    def __init__(self):
        super().__init__()
        self._add_train_args()
        self.args = self.parser.parse_args()

    def _add_train_args(self):
        # Train args
        self.parser.add_argument('--device', type=str, default='cpu', help="CUDA device for training")
        self.parser.add_argument('--algo', type=str, default='PPO', help="RL algorithm to use")
        self.parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes to rollout")

        # CityLearn config
        self.parser.add_argument('--frac', type=float, default=1.0, help="Fraction of days in the dataset to use for training")
        self.parser.add_argument('--render', action='store_true', help="Flag for using `CityLearnEnv.render()`")
        self.parser.add_argument('--custom', action='store_true', help="Flag for CityLearn dataset customization")

        # Logging
        self.parser.add_argument('--name', type=str, nargs='?', help="Experiment name (used for directory)")
        self.parser.add_argument('--wandb', action='store_true', help="Flag for logging on wandb")
        self.parser.add_argument('--entity', type=str, default='universitaverona', help="Wandb entity")
        self.parser.add_argument('--project', type=str, default='citylearn_omnisafe', help="Wandb project name")
        self.parser.add_argument('--tag', type=str, default='comfort_reward', help="Wandb tag")

    def save_yaml(self, dir: str):
        with open(f'{dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)


class EvalConfig(Config):
    def __init__(self):
        super().__init__()
        self._add_eval_args()
        self.args = self.parser.parse_args()

    def _add_eval_args(self):
        # Eval args
        self.parser.add_argument('--exp_dir', type=str, default='./experiments/PPO_seed1_04-11-25_14:46:25', help="Path to the experiment of the RL agent to evaluate")
        self.parser.add_argument('--test', action='store_true', help="Evaluation mode")

        # CityLearn config
        self.parser.add_argument('--building', type=int, default=2, help="Whether to evaluate on the same building of training")