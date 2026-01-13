# Omnisafe Wrapper for CityLearn Environment

## Environment Setup

1. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

2. Setup `omnisafe`
```console
# Create conda environment
cd omnisafe
conda env create --file conda-recipe.yaml

# Activate and setup omnisafe
conda activate omnisafe
pip install -e .

# Update `torch` and `torchvision`
pip install torch==2.8.0
pip install torchvision==0.23.0
``` 

3. Setup `citylearn`:
```console
# From main direcotry
cd CityLearn
pip install -e .
```

4. Additional requirements
```console
# Progress bar visualization
pip install ipywidgets

# SB3 comparison
pip install stable-baselines3==2.0.0
```

## Running Experiments

### Training

To train an agent, run the `train.py` script located in the `scripts/` directory.

```bash
python scripts/train.py --algo PPO --episodes 1000
````

#### Logging with Weights & Biases

To log training metrics to Weights & Biases, enable the `--wandb` flag and specify your W&B entity (typically your username):

```bash
python scripts/train.py \
  --algo PPO \
  --episodes 1000 \
  --wandb \
  --entity <your_wandb_entity>
```

#### Train / Validation Split

To enable a train–validation split, use the `--frac` flag to specify the proportion of data used for training (e.g., `0.8`, `0.7`):

```bash
python scripts/train.py \
  --algo PPO \
  --episodes 1000 \
  --frac 0.8
```

#### Algorithm Selection and Training Length

Choose one of the supported OmniSafe algorithms using the `--algo` flag (e.g., `PPO`, `SAC`) and set the number of training episodes with `--episodes`:

```bash
python scripts/train.py --algo PPO --episodes 1000
```

---

### Modifying Training Parameters

To customize algorithm-specific training parameters:

1. Locate the corresponding OmniSafe configuration file in:

   ```
   omnisafe/omnisafe/configs/
   ```

   For example:

   ```
   omnisafe/omnisafe/configs/on-policy/PPO.yaml
   ```

2. Review the available parameters under the `algo_cfgs` section of the config file.

3. In `train.py`, add or override the desired parameters in the `custom_cfgs` dictionary under the `algo_cfgs` key.

---

## Customizing the Reward Function

By default, if no reward is specified, the environment uses the **default CityLearn reward defined in the schema** (`ComfortReward`).

### Selecting a Reward Function via Command Line

To use a different reward function, pass the `--tag` argument when running `train.py`.

Example:

```bash
python scripts/train.py \
  --algo PPO \
  --episodes 1000 \
  --tag solar_comfort_reward
```

The value passed to `--tag` must correspond to a key in the `CUSTOM_REWARD_FN` dictionary defined in `train.py`.

---

### Available Reward Functions

All selectable reward functions are registered in the `CUSTOM_REWARD_FN` dictionary inside `train.py`.
Each entry maps a tag name to a CityLearn reward function class.

Example:

```python
CUSTOM_REWARD_FN = {
    'solar_comfort_reward': {
        'type': 'citylearn.reward_function.SolarPenaltyAndComfortReward'
    }
}
```

---

### Adding a New Reward Function

To add a new reward function, follow these steps:

1. **Select an existing reward function**

   Choose a reward function that already exists in:

   ```
   CityLearn/citylearn/reward_function.py
   ```

   For example, to use `SolarPenaltyAndComfortReward`, ensure that the class is defined in this file.

2. **Register the reward in `train.py`**

   Add a new entry to the `CUSTOM_REWARD_FN` dictionary that points to the selected reward function:

   ```python
   CUSTOM_REWARD_FN['solar_comfort_reward'] = {
       'type': 'citylearn.reward_function.SolarPenaltyAndComfortReward'
   }
   ```

3. **Select the reward during training**

   Specify the corresponding tag using the `--tag` argument:

   ```bash
   python scripts/train.py \
     --algo PPO \
     --episodes 1000 \
     --tag solar_comfort_reward
   ```

---

### Defining a Custom Reward Function (Optional)

If the reward function you want to use is **not already available** in `CityLearn/citylearn/reward_function.py`, you can define a new custom reward class there.

When implementing a new reward function, please follow the official CityLearn guidelines to ensure compatibility:

👉 [https://www.citylearn.net/overview/reward_function.html#how-to-define-a-custom-reward-function](https://www.citylearn.net/overview/reward_function.html#how-to-define-a-custom-reward-function)

After defining the new reward class, register it in `CUSTOM_REWARD_FN` and select it via the `--tag` argument as described above.

---

### Evaluation

To evaluate a trained agent, use the `evaluate.py` script.

You must specify the experiment directory containing the trained model using the `--exp_dir` flag:

```bash
python scripts/evaluate.py \
  --exp_dir ./experiments/PPO_seed1_04-11-25_14:46:25
```

#### Evaluation Modes

* **Validation (default):**
  Evaluates the agent on the validation split (held-out portion of the training data).

* **Test:**
  To evaluate on the test set (same dataset as training but a different building), enable the `--test` flag:

```bash
python scripts/evaluate.py \
  --exp_dir ./experiments/PPO_seed1_04-11-25_14:46:25 \
  --test
```

## Our modifications

### a. Omnisafe
Current modifications applied to `omnisafe`:

1. `omnisafe/omnisafe/envs/citylearn_env.py` -> implemented the omnisafe wrapper for CityLearn

2. `ominsafe/omnisafe/configs` -> added an empty dictionary `env_cfgs` under `default` configurations in yaml files for enabling external environments wrapping

3. `omnisafe/omnisafe/envs/wrapper.py` (rows 510-525) -> modified `ActionScale.step()` so that it does not systematically map continuous actions in [-1, 1]

4. `omnisafe/omnisafe/models/actor/gaussian_learning_actor.py` (row 98-102) -> return `torch.tanh(action)` for consistent action scaling when using on-policy algos

### b. CityLearn
Current modifications applied to `citylearn`:

1. `CityLearn/citylearn/reward_function.py` (rows 526-727) -> implemented `Challenge2023Reward` and `Challenge2023ComfortReward`:
```python
class Challenge2023ComfortReward(ComfortReward):
    """
    Reward representing CityLearn Challenge 2023 COMFORT control score 
    (see at https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge/problems/control-track-citylearn-challenge). 

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    weights: Mapping[str ,float]:
        Weights weighting different portion of the control score listed in the challenge.
    band: float, default: 2.0
        Setpoint comfort band (+/-). If not provided, the comfort band time series defined in the
        building file, or the default time series value of 2.0 is used.
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.

    NOTE
    ----------
    This reward function differ to its super class in taking into account only situations in which `occupant_count` > 0 for a building.
    Moreover, depending on `power_outage` signal, the reward is differently weighted following weights values listed in the challenge.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], weights: Mapping[str, float], band: float = None, lower_exponent: float = None, higher_exponent: float = None):
        super().__init__(env_metadata, band, higher_exponent, lower_exponent)
        self.weights = weights

    def calculate(self, observations: List[Mapping[str, Union[int, float]]], stand_alone: bool = True) -> List[float]:
        reward_list = []

        for o in observations:
            reward = 0.0

            if o['occupant_count'] > 0:
                heating_demand = o.get('heating_demand', 0.0)
                cooling_demand = o.get('cooling_demand', 0.0)
                heating = heating_demand > cooling_demand
                hvac_mode = o['hvac_mode']
                indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']

                if hvac_mode in [1, 2]:
                    set_point = o['indoor_dry_bulb_temperature_cooling_set_point'] if hvac_mode == 1 else o['indoor_dry_bulb_temperature_heating_set_point']
                    band =  self.band if self.band is not None else o['comfort_band']
                    lower_bound_comfortable_indoor_dry_bulb_temperature = set_point - band
                    upper_bound_comfortable_indoor_dry_bulb_temperature = set_point + band
                    delta = abs(indoor_dry_bulb_temperature - set_point)
                    
                    if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                        exponent = self.lower_exponent if hvac_mode == 2 else self.higher_exponent
                        reward = -(delta**exponent)
                    
                    elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                        reward = 0.0 if heating else -delta

                    elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                        reward = -delta if heating else 0.0

                    else:
                        exponent = self.higher_exponent if heating else self.lower_exponent
                        reward = -(delta**exponent)

                else:
                    cooling_set_point = o['indoor_dry_bulb_temperature_cooling_set_point']
                    heating_set_point = o['indoor_dry_bulb_temperature_heating_set_point']
                    band =  self.band if self.band is not None else o['comfort_band']
                    lower_bound_comfortable_indoor_dry_bulb_temperature = heating_set_point - band
                    upper_bound_comfortable_indoor_dry_bulb_temperature = cooling_set_point + band
                    cooling_delta = indoor_dry_bulb_temperature - cooling_set_point
                    heating_delta = indoor_dry_bulb_temperature - heating_set_point

                    if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                        exponent = self.higher_exponent if not heating else self.lower_exponent
                        reward = -(abs(heating_delta)**exponent)

                    elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < heating_set_point:
                        reward = -(abs(heating_delta))

                    elif heating_set_point <= indoor_dry_bulb_temperature <= cooling_set_point:
                        reward = 0.0

                    elif cooling_set_point < indoor_dry_bulb_temperature < upper_bound_comfortable_indoor_dry_bulb_temperature:
                        reward = -(abs(cooling_delta))

                    else:
                        exponent = self.higher_exponent if heating else self.lower_exponent
                        reward = -(abs(cooling_delta)**exponent)

            # Different weights depending on the poware outage signal 
            outage_signal = o.get('power_outage', False)
            if not outage_signal:
                reward *= self.weights['w1']
            else:
                reward *= self.weights['w4']

            reward_list.append(reward)

        if self.central_agent:
            if not stand_alone: # if called by `Challenge2023Reward`
                reward = reward_list
            else:
                reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward
    

class Challenge2023Reward(RewardFunction):
    """
    Reward tackling CityLearn Challenge 2023 TOTAL control score 
    (see at https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge/problems/control-track-citylearn-challenge). 

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    weights: Mapping[str ,float]:
        Weights weighting different portion of the control score listed in the challenge.
    band: float, default: 2.0
        Setpoint comfort band (+/-). If not provided, the comfort band time series defined in the
        building file, or the default time series value of 2.0 is used.
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    """

    def __init__(
            self, 
            env_metadata: Mapping[str, Any], 
            weights: Mapping[str, float],
            band: float = None, 
            lower_exponent: float = None, 
            higher_exponent: float = None
        ):
        super().__init__(env_metadata)

        # Phase weights
        self.weights = weights

        # Neighborhood net elecrticity consumption info
        self.E_t_prev = 0.0
        self.E_t_max = 0.0

        # Comfort score function
        self.__comfort_fn = Challenge2023ComfortReward(
            env_metadata=env_metadata,
            weights=weights,
            band=band, 
            higher_exponent=higher_exponent, 
            lower_exponent=lower_exponent
        )

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]):

        # Check for metadata in `__comfort_fn`
        if self.__comfort_fn.env_metadata is None:
            self.__comfort_fn.env_metadata = self.env_metadata 

        # Comfort score
        comfort_reward = self.__comfort_fn.calculate(observations, stand_alone=False)
        # Current neighborhood net electricity consumption
        E_t = 0.0

        reward_list = []
        for i, o in enumerate(observations):
            reward = 0.0

            # Carbon emissions score
            e_t = o['net_electricity_consumption']
            B_t = o['carbon_intensity']
            reward = comfort_reward[i] - self.weights['w2']*(e_t*B_t) 

            # Update neighborhood net consumption
            E_t += e_t

            reward_list.append(reward)

        # Neighborhood net electricity consumption info update
        ramping = abs(E_t - self.E_t_prev)
        self.E_t_prev = E_t
        nh_max_consumption = max(0.0, E_t-self.E_t_max)
        self.E_t_max = max(E_t, self.E_t_max)

        if self.central_agent:
            reward = sum(reward_list)

            # Grid control score      
            reward += self.weights['w3']*(nh_max_consumption - ramping)
            reward = [reward]
        
        else:
            reward = reward_list

        return reward
```