# CityLearn
from citylearn.agents.base import Agent
from citylearn.agents.rbc import BasicRBC, HourRBC
from citylearn.building import Building
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import OptimizedRBC

# Utils
import torch
import numpy as np
from torch.distributions import Normal
from typing import Any, List, Mapping, Union


class ComfortRBC(OptimizedRBC):
    """
    Rule-based Control designed to overwrite controls scheduled by :py:class:`citylearn.agents.rbc.OptimizedRBC` 
    in order to tackle temperature discomfort.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment to perform control on.
    band: float
        Comfort band to try to satisfy. 

    TODO
    ---------- 
    Understand how to manage storages and devices with respect to them
    """
    def __init__(self, env: CityLearnEnv, band: float=None, **kwargs):        
        # Init OptimizedRBC
        super().__init__(env, **kwargs)

        # Sanity check
        self._check(env)

        # Comfort band (+/-) to satisfy
        self.comfort_band = band if band is not None else env.buildings[0].comfort_band[0] 

    def predict(self, observations: List[List[float]], deterministic: bool=None) -> List[List[float]]:        
        # Predict actions based on hour scheduling
        scheduled_acions = super().predict(observations, deterministic)

        actions = []
        for i, o in enumerate(observations):
            action = scheduled_acions[i]

            # Available spaces
            available_obs = self.observation_names[i]
            available_act = self.action_names[i]

            # Temperatures
            if 'indoor_dry_bulb_temperature' in available_obs:
                indoor_temp = o[available_obs.index('indoor_dry_bulb_temperature')]
            else:
                indoor_temp = None

            if 'outdoor_dry_bulb_temperature' in available_obs:
                outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature')]
            else:
                outdoor_temp = None

            if 'indoor_dry_bulb_temperature_cooling_set_point' in available_obs:
                cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]
            else:
                cooling_setpoint = None

            if 'indoor_dry_bulb_temperature_cooling_delta' in available_obs:
                cooling_delta = o[available_obs.index('indoor_dry_bulb_temperature_cooling_delta')]
            else:
                cooling_delta = None

            if 'indoor_dry_bulb_temperature_heating_set_point' in available_obs:
                heating_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_heating_set_point')]
            else:
                heating_setpoint = None

            if 'indoor_dry_bulb_temperature_heating_delta' in available_obs:
                heating_delta = o[available_obs.index('indoor_dry_bulb_temperature_heating_delta')]
            else:
                heating_delta = None

            # Stoarges SoC
            if 'electrical_storage_soc' in available_obs:
                electrical_soc = o[available_obs.index('electrical_storage_soc')]
            else:
                electrical_soc = -1

            if 'cooling_storage_soc' in available_obs:
                cooling_soc = o[available_obs.index('cooling_storage_soc')]
            else:
                cooling_soc = -1

            if 'heating_storage_soc' in available_obs:
                heating_soc = o[available_obs.index('heating_storage_soc')]
            else:
                heating_soc = -1

            # Manage cooling
            if 'cooling_device' in available_act:
                # Action indexes
                device_idx = available_act.index('cooling_device')
                if 'electircal_storage' in available_act:
                    ess_idx = available_act.index('electrical_storage')
                else:
                    ess_idx = None

                # Temperature difference
                hot_delta = cooling_delta if cooling_delta is not None else indoor_temp - cooling_setpoint
                if hot_delta > 0:
                    if hot_delta > self.comfort_band: # Too hot -> supply the cooling device                        
                        action[device_idx] = 0.8
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/2)
                    else:
                        action[device_idx] = 0.2 # Hot within the band
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/3)
                else:
                    if indoor_temp is not None and outdoor_temp is not None:
                        temp_delta = outdoor_temp - indoor_temp # Outdoor temperature affects indoor temperature                       
                        action[device_idx] = 0.3 if temp_delta > 0 else 0.0

                    else:
                        action[device_idx] = 0.0
                        if ess_idx is not None: 
                            action[ess_idx] = action[ess_idx]/2 if action[ess_idx] < 0 else action[ess_idx]

            # Manage heating
            if 'heating_device' in available_act:
                # Action indexes
                device_idx = available_act.index('heating_device')
                if 'electrical_storage' in available_act:
                    ess_idx = available_act.index('electrical_storage')
                else:
                    ess_idx = None

                # Temperature difference
                cold_delta = heating_delta if heating_delta is not None else indoor_temp - heating_setpoint
                if cold_delta < 0:
                    if cold_delta < -self.comfort_band:
                        action[device_idx] = 0.8 # Too cold -> supply the heating device
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/2)
                    else:
                        action[device_idx] = 0.2 # Cold within the band
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/3)
                else:
                    if indoor_temp is not None and outdoor_temp is not None:
                        temp_delta = outdoor_temp - indoor_temp # Outdoor temperature affects indoor temperature
                        action[device_idx] = 0.3 if temp_delta < 0 else 0.0
                    else:
                        action[device_idx] = 0.0
                        if ess_idx is not None: 
                            action[ess_idx] = action[ess_idx]/2 if action[ess_idx] < 0 else action[ess_idx]

            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions
    
    def _check(self, env: CityLearnEnv):
        if 'indoor_dry_bulb_temperature' in env.observation_names[0]:
            if 'indoor_dry_bulb_temperature_cooling_set_point' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_set_point' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_cooling_delta' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_delta' not in env.observation_names[0]:
                raise RuntimeError(
                    '`indoor_dry_bulb_temperature` is available, but no `indoor_dry_bulb_temperature_*_set_point` ' +
                    'or  `indoor_dry_bulb_temperature_*_delta` is available.'
                )
        else:
            if 'indoor_dry_bulb_temperature_cooling_delta' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_delta' not in env.observation_names[0]:
                raise RuntimeError('No `indoor_dry_bulb_temperature_*_delta` is available.')


class OmnisafeActorWrapper:
    """
    Wrapper for loading and using an Omnisafe actor for evaluation.

    Parameters
    ----------
    fname: str
        Path to the jit script of the actor's network.
    env: CityLearnEnv
        CityLearn environment to perform evaluation on.
    """

    def __init__(self, fname: str, env: CityLearnEnv):
        
        # Load actor net from file
        self.actor = torch.jit.load(fname)
        # CityLearn env action space for action scaling
        self.output_space = env.action_space[0]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # Unwrap CityLearn observations
        obs = torch.as_tensor(obs[0], dtype=torch.float32)

        # Action distribution for given observation
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        distr = Normal(mean, std)

        # Predicted action
        action = torch.tanh(distr.mean)
        action = action.detach().cpu().numpy()

        # Scale action to CityLearn range
        action = self._scale(action)

        return action[np.newaxis, ...]
    
    def _scale(self, action: np.ndarray) -> np.ndarray:
        # Input space
        input_high = np.ones_like(self.output_space.high)
        input_low = -np.ones_like(self.output_space.low)
        input_range = input_high - input_low

        # Scale action to desired range
        output_range = self.output_space.high - self.output_space.low
        scaled_vector = (action - input_low) / input_range
        action = self.output_space.low + scaled_vector * output_range

        return action
    
class CustomRBC(BasicRBC):

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

    @HourRBC.action_map.setter
    def action_map(self, action_map: Union[List[Mapping[str, Mapping[int, float]]], Mapping[str, Mapping[int, float]], Mapping[int, float]]):
        if action_map is None:
            action_map = {}
            action_names = [a_ for a in self.action_names for a_ in a]
            action_names = list(set(action_names))


            for n in action_names:
                action_map[n] = {}

                if 'electrical_storage' in n:
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        # TODO: Implement RBC policy

                        action_map[n][hour] = value
                
                elif n == 'dhw_storage':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        # TODO: Implement RBC policy

                        action_map[n][hour] = value

                elif n == 'cooling_device':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        # TODO: Implement RBC policy

                        action_map[n][hour] = value
                
                else:
                    raise ValueError(f'Unknown action name: {n}')
                
        # Imposta la mappa nella superclasse
        HourRBC.action_map.fset(self, action_map)

class AdvancedRBC(Agent):
    """
    Advanced Rule-Based Controller (RBC) Agent with comfort band consideration.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment to perform control on.
    band: float
        Comfort band to try to satisfy. 

    """
    def __init__(self, env: CityLearnEnv, band: float=2.0, **kwargs):

        # Init OptimizedRBC
        super().__init__(env, **kwargs)

        # Comfort band (+/-) to satisfy
        self.comfort_band = band 

    def predict(self, observations: List[List[float]], deterministic: bool = True) -> List[List[float]]:        

        actions = []
        for i, o in enumerate(observations):

            # Available spaces
            available_obs = self.observation_names[i]
            available_act = self.action_names[i]
            action = [0.0 for _ in range(len(available_act))]

            # TODO add other observations if needed
            # TODO implement more advanced RBC logic for each device

            # Indoor temperature and setpoints
            indoor_temp = o[available_obs.index('indoor_dry_bulb_temperature')]
            cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]


            if 'cooling_device' in available_act:
                # EXAMPLE LOGIC: Turn on cooling if indoor temp exceeds setpoint + comfort band
                if indoor_temp > cooling_setpoint + self.comfort_band:
                    action[available_act.index('cooling_device')] = 1.0  # Turn on cooling
                else:
                    action[available_act.index('cooling_device')] = 0.0  # Turn off cooling

            if 'electrical_storage' in available_act:
                pass

            if 'dhw_storage' in available_act:
                pass

            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions