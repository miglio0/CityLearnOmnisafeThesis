# CityLearn utils
from citylearn.data import DataSet
from citylearn.wrappers import StableBaselines3Wrapper

# Logging
import wandb
from stable_baselines3.common.callbacks import BaseCallback

# Utils
import argparse
from collections import deque
from typing import Any, List, Tuple, Dict


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Configurations for Constrained RL on CityLearn")
        self._add_args()
        self.args = self.parser.parse_args()

    def _add_args(self):
        # Seed
        self.parser.add_argument('--seed', type=int, default=1, help="Experiment seed")

        # CityLearn config
        self.parser.add_argument('--data', type=str, default='citylearn_challenge_2023_phase_1', help="CityLearn dataset")
        self.parser.add_argument('--custom', action='store_true', help="Flag for CityLearn dataset customization")

        # RL args
        self.parser.add_argument('--wrapper', type=str, choices=['omnisafe', 'sb3'], default='omnisafe', help="CityLearn wrapper to use")
        self.parser.add_argument('--algo', type=str, default='PPO', help="RL algorithm to use")
        self.parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes to rollout")

        # Logging
        self.parser.add_argument('--wandb', action='store_true', help="Flag for logging on wandb")
        self.parser.add_argument('--entity', type=str, default='luca0', help="Wandb entity")
        self.parser.add_argument('--project', type=str, default='citylearn_omnisafe', help="Wandb project name")
        self.parser.add_argument('--tag', type=str, default='comfort_reward', help="Wandb tag")


class CityLearnKPIWrapper(StableBaselines3Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            # Get KPIs
            kpis = self.env.evaluate()

            # Filter district level KPIs
            kpis = kpis[kpis['level'] == 'district']
            discomfort = kpis[(kpis['cost_function'] == 'discomfort_cold_delta_average')]['value'].item()
            carbon_emissions = kpis[(kpis['cost_function'] == 'carbon_emissions_total')]['value'].item()
            net_consumption = kpis[(kpis['cost_function'] == 'electricity_consumption_total')]['value'].item()

            # Populate info dict
            info['discomfort'] = discomfort
            info['carbon_emissions'] = carbon_emissions
            info['net_consumption'] = net_consumption

        return obs, reward, terminated, truncated, info


class CityLearnWandbCallback(BaseCallback):
    def __init__(self, verbose: int=0, window_len: int=100):
        super().__init__(verbose)
        
        # Episodic info
        self.ep_count = 0
        self.ep_rewards = deque(maxlen=window_len)
        self.ep_lengths = deque(maxlen=window_len)

        # KPIs info
        self.discomfort_h = deque(maxlen=window_len)
        self.carbon_emissions_h = deque(maxlen=window_len)
        self.net_consumption_h = deque(maxlen=window_len)

        # Log fn
        self._log_fn = lambda x: sum(x)/len(x)

    def _on_step(self) -> bool:
        
        info = self.locals['infos'][0]
        if self.locals['dones'][0]:
            # Update episode count
            self.ep_count += 1

            # Episodic info
            ep_info = info['episode']
            self.ep_rewards.append(ep_info['r'])
            self.ep_lengths.append(ep_info['l'])

            # KPIs
            self.discomfort_h.append(info['discomfort'])
            self.carbon_emissions_h.append(info['carbon_emissions'])
            self.net_consumption_h.append(info['net_consumption'])
                
            # Log episodic return and length to wandb
            wandb.log(
                {
                    'TotalEnvSteps': self.num_timesteps,
                    'Metrics/Discomfort': self._log_fn(self.discomfort_h),
                    'Metrics/CO2_Emissions': self._log_fn(self.carbon_emissions_h),
                    'Metrics/Electricity_Consumption': self._log_fn(self.net_consumption_h),
                    'Metrics/EpRet': self._log_fn(self.ep_rewards),
                    'Metrics/EpLen': self._log_fn(self.ep_lengths),
                    'Metrics/EpCost': 0.0
                },
                step=self.ep_count
            )

            print(
                f"{'*'*30}\nEPISODE {self.ep_count}"                          +
                f'\n- Discomfort:              {self.discomfort_h[-1]}'       +
                f'\n- CO2 Emissions:           {self.carbon_emissions_h[-1]}' +
                f'\n- Electricity Consumption: {self.net_consumption_h[-1]}'  +
                f'\n- Reward:                  {self.ep_rewards[-1]}'         +
                f'\n- Length:                  {self.ep_lengths[-1]}'
            )
        
        return True
    

def _select_items(schema: Dict[str, Any], key: str, available_items: List[str]=None) -> Tuple[dict, List[str]]:
    assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'
    
    # Init
    flag_key = 'include' if key == 'buildings' else 'active'
    pool = available_items if available_items is not None else list(schema[key].keys())

    print(f"Available {key}:")
    for idx, item in enumerate(pool):
        print(f"- {idx+1}. {item}")

    # Item selection
    user_input = input(f"\nSelect {item} by entering their numbers separated by commas (e.g., 1,3,5): ")
    selected_indices = [int(i.strip()) - 1 for i in user_input.split(',') if i.strip().isdigit() and 0 < int(i.strip()) <= len(pool)]
    selected_items = [pool[i] for i in selected_indices]

    print(f"Selected items: {selected_items}\n\n")

    # Filter CityLearn items based on user selection
    for item in schema[key].keys():
        schema[key][item][flag_key] = (item in selected_items) 

    return schema, selected_items
    

def select_env_config(dataset: str) -> Dict[str, Any]:

    print("="*40)
    print("CityLearnOmnisafe")
    print("="*40)
    print(f"Dataset: {dataset}")

    # Filter schema's available observations/actions 
    schema = DataSet().get_schema(dataset)
    available_obs = [obs for obs in schema['observations'].keys() if schema['observations'][obs]['active']]
    available_act = [act for act in schema['actions'].keys() if schema['actions'][act]['active']]

    # User's building selection
    schema, _ = _select_items(schema=schema, key='buildings')
    # User's observation selection
    schema, selected_obs = _select_items(schema=schema, key='observations', available_items=available_obs)
    # User's action selection
    schema, selected_act = _select_items(schema=schema, key='actions', available_items=available_act)

    for action in selected_act:
        device = action.split('_')[0]
        
        if not any(device in obs for obs in selected_obs):
            print(f"[WARN] You selected to control '{action}', but no observation related to '{device}' is selected. Please select again.")

    return schema


def default_env_config(dataset: int) -> Dict[str, Any]:
    # Get schema from CityLearn dataset
    schema = DataSet().get_schema(dataset)

    # Our default configuration considers only `Building_1`
    for build in schema['buildings'].keys():
        schema['buildings'][build]['include'] = (build == 'Building_1')

    return schema