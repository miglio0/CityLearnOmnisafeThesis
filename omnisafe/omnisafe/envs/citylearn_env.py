# Omnisafe
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

# CityLearn
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

# Utils
import torch
from typing import Any, ClassVar


@env_register
class CityLearnOmnisafe(CMDP):
    _support_envs: ClassVar[list[str]] = ['CityLearn-v0']

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(
        self,
        env_id: str,
        device: torch.device=DEVICE_CPU,
        **kwargs
    ):
        super().__init__(env_id)
        
        # Required attrs
        self._num_envs = 1
        self.type = 'CityLearn'

        # Env info
        self._env = NormalizedObservationWrapper(CityLearnEnv(**kwargs))
        self._observation_space = self._env.observation_space[0]
        self._action_space = self._env.action_space[0]

        # Device
        self._device = device

    @property
    def max_episode_steps(self):
        return self._env.time_steps - 1

    def reset(
        self,
        seed: int | None=None,
        options: dict[str, Any] | None=None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        
        # Reset the wrapped environment
        obs, info = self._env.reset(seed=seed, options=options)
        obs = obs[0]
       
        # Convert observations to torch tensor
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info
    
    def step(self, action: torch.Tensor):

        # Reshape actions to fit CityLearn
        action = action.unsqueeze(0).detach().cpu().numpy()
        
        # Perform `.step()` in the wrapped env
        obs, reward, terminated, truncated, info = self._env.step(action)

        if terminated:
            # Get KPIs
            kpis = self._env.evaluate()

            # Filter district level KPIs
            kpis = kpis[kpis['level'] == 'district']
            discomfort = kpis[(kpis['cost_function'] == 'discomfort_cold_delta_average')]['value'].item()
            carbon_emissions = kpis[(kpis['cost_function'] == 'carbon_emissions_total')]['value'].item()
            net_consumption = kpis[(kpis['cost_function'] == 'electricity_consumption_total')]['value'].item()

            # Populate info dict
            info['discomfort'] = torch.as_tensor(discomfort, dtype=torch.float32, device=self._device)
            info['carbon_emissions'] = torch.as_tensor(carbon_emissions, dtype=torch.float32, device=self._device)
            info['net_consumption'] = torch.as_tensor(net_consumption, dtype=torch.float32, device=self._device)
   
        # Retrieve `obs` and `reward` as torch tenors
        obs = torch.as_tensor(obs[0], dtype=torch.float32, device=self._device)
        reward = torch.as_tensor(reward[0], dtype=torch.float32, device=self._device)

        # Convert `truncated` and `terminated` into torch tensors
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self._device)
        truncated = torch.as_tensor(truncated, dtype=torch.bool, device=self._device)        

        # Placeholder
        cost = torch.zeros_like(reward)

        return obs, reward, cost, terminated, truncated, info
    
    def set_seed(self, seed: int) -> None:
        self.reset(seed=seed)

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
