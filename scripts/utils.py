# CityLearn utils
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.wrappers import StableBaselines3Wrapper

# Logging
import wandb
from stable_baselines3.common.callbacks import BaseCallback

# Utils
import json
from collections import deque
from typing import Any, List, Tuple, Dict


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
    def __init__(self, verbose: int=0, online: bool=False, window_len: int=100):
        super().__init__(verbose)

        # Whether to log data online
        self.online = online
        
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
            if self.online:
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
    

class CityLearnSchema:
    def __init__(self, schema: Dict[str, Any] | None=None):
        self._schema: Dict[str, Any] | None = schema

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema
    
    @schema.setter
    def schema(self, new_schema: Dict[str, Any]):
        self._schema = new_schema

    def load(self, dataset: str, custom: bool=False):
        assert self._schema is None, 'Schema has already been loaded.'

        # Get the schema of the dataset
        self._schema = DataSet().get_schema(dataset)

        if custom:
            print("="*40)
            print("CityLearnOmnisafe - Dataset Customization")
            print("="*40)
            print(f"Dataset: {dataset}")

            # User's building selection
            _ = self._select_items(key='buildings')
            # User's observation selection
            selected_obs = self._select_items(key='observations')
            # User's action selection
            selected_act = self._select_items(key='actions')

            # Sanity check
            self._check(selected_obs, selected_act)
        else:
            # Include Building_1 by default
            self.set_active(key='buildings', items=['Building_1'])

        # Custom cooling device: Daikin FTXM35R-RXM35R
        # page 44 https://planetaklimata.com.ua/instr/Daikin/Daikin_Perfera_RXM-R_Data_Sheet_Eng.pdf
        # for _, b in self._schema['buildings'].items():
        #     b['cooling_device'] = {
        #         'type': 'citylearn.energy_model.AirConditioner',
        #         'autosize': False,
        #         'attributes': {
        #             'nominal_power': 3.40,
        #             'nominal_efficiency': 4.04,
        #             'rated_outdoor_temperature': 35.0,
        #             'efficiency_derating': 0.12
        #         }
        #     }

    def save(self, dir: str, prefix: str='base'):
        with open(f'{dir}/{prefix}_schema.json', 'w') as f:
            json.dump(self._schema, f, indent=4)

    def set(self, key: str, value: Dict[str, Any]):
        self._schema[key] = value

    def set_active(self, key: str, items: List[str]):
        assert self._schema is not None, 'Schema has not been loaded, yet.'
        assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'

        # Filter CityLearn items
        flag_key = 'include' if key == 'buildings' else 'active'
        for it in self._schema[key].keys():
            self._schema[key][it][flag_key] = (it in items)

    def train_test_split(self, frac: float, mode: str):
        assert mode in ['train', 'test'], f'Unknown mode {mode}. Must be either `train` or `test`.'
        assert 0 < frac <= 1, f'Invalid fraction {frac}. Must be in (0,1).'

        # Copy base schema
        train_schema, test_schema = self._schema.copy(), self._schema.copy()

        # Total simulation days
        time_steps = self._schema['simulation_end_time_step'] + 1
        total_days = time_steps // 24

        # Train/test split index
        train_days = int(total_days * frac)
        split_idx = train_days * 24

        # Modify train/test schemas
        train_schema['simulation_end_time_step'] = split_idx - 1
        if frac < 1:
            test_schema['simulation_start_time_step'] = split_idx

        return train_schema, test_schema

    def _select_items(self, key: str):
        assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'
    
        # Available items
        if key == 'buildings':
            pool = list(self._schema[key].keys())
        else:
            pool = [item for item in self._schema[key].keys() if self._schema[key][item]['active']]

        print(f"Available {key}:")
        for idx, item in enumerate(pool):
            print(f"- {idx+1}. {item}")

        # Item selection
        user_input = input(f"\nSelect {item} by entering their numbers separated by commas (e.g., 1,3,5): ")
        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',') if i.strip().isdigit() and 0 < int(i.strip()) <= len(pool)]
        selected_items = [pool[i] for i in selected_indices]

        print(f"Selected items: {selected_items}\n\n")

        # Modify schema according to user's selection
        self.set_active(key=key, items=selected_items)

        return selected_items
    
    def _check(self, observations: List[str], actions: List[str]):
        print('Checking observations...')
        if 'indoor_dry_bulb_temperature' in observations:
            if 'indoor_dry_bulb_temperature_cooling_set_point':
                # Remove "redundant" observations
                observations.remove('indoor_dry_bulb_temperature_cooling_set_point')

                # Activate temperature delta
                observations.append('indoor_dry_bulb_temperature_cooling_delta')
                self.set_active(key='observations', items=observations)
                print(
                    '[CHECK] Both `indoor_dry_bulb_temperature` and `indoor_dry_bulb_temperature_cooling_set_point` are active.' + 
                    ' `indoor_dry_bulb_temperature_cooling_delta` has been activated.'
                )

            if 'indoor_dry_bulb_temperature_heating_set_point' in observations:
                # Remove "reduntant" observations
                observations.remove('indoor_dry_bulb_temperature_heating_set_point')

                # Activate temperature delta
                observations.append('indoor_dry_bulb_temperature_heating_delta')
                self.set_active(key='observations', items=observations)
                print(
                    '[CHECK] Both `indoor_dry_bulb_temperature` and `indoor_dry_bulb_temperature_heating_set_point` are active.' + 
                    ' `indoor_dry_bulb_temperature_heating_delta` has been activated.'
                )


def get_kpis(env: CityLearnEnv) -> Dict[str, float]:
    kpis = env.unwrapped.evaluate()

    # KPIs to retrieve
    kpi_names = {
        'cost_total': 'Cost ($/kWh)',
        'carbon_emissions_total': 'Emissions (kgC02e/kWh)',
        'daily_peak_average': 'Avg. daily peak (kWh)',
        'ramping_average': 'Ramping (kWh)',
        'daily_one_minus_load_factor_average': '1 - load factor',
        'discomfort_proportion': 'Discomfort (%)'
    }

    # Filter KPIs
    kpis = kpis[
        (kpis['level'] == 'district') &
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()
    kpis['cost_function'] = kpis['cost_function'].map(lambda x: kpi_names[x])

    kpis_dict = {}
    for _, kpi in kpis.iterrows():
        kpis_dict[kpi['cost_function']] = kpi['value']

    return kpis_dict

def placeholder_cost_fn(obs: torch.Tensor) -> torch.Tensor:
    # Placeholder cost function that returns zero cost
    return torch.tensors([0.0], dtype=torch.float32)