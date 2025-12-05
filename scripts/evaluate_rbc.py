# Our agents
from agent import *

# CityLearn utils
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.agents.rbc import OptimizedRBC

# Utils
import os
import json, yaml
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from config import EvalConfig
from collections import OrderedDict, defaultdict
from utils import *




def plot_battery_h(history: Dict[str, List[float]], ax1: Axes):
    assert 'soc' in history.keys(), 'Missing state of charge information in battery history.'
    assert 'discharge' in history.keys(), 'Missing charge rate information in battery history.'
    ax1.set_axisbelow(True)
    ax1.grid(visible=True, linestyle='dashed')

    # Charge rate
    ax1.bar(range(len(history['discharge'])), history['discharge'], color='xkcd:soft blue')
    ax1.set_ylabel('(Dis)Charge (kW/h)')
    ax1.yaxis.label.set_color('xkcd:soft blue')

    ax2 = ax1.twinx()

    # State of charge
    ax2.plot(history['soc'], c='xkcd:orange')
    ax2.set_ylabel('SoC (%)')
    ax2.set_ylim(ymin=-0.05, ymax=1.05)
    ax2.yaxis.label.set_color('xkcd:orange')


def compare_temperature(args, res_1, res_2, algo_names=[]):
    # RBC temperature
    temp_1 = res_1['env_h']['temperature']
    temp_2 = res_2['env_h']['temperature']

    
    fig = plt.figure(figsize=[30,10])
    fig.suptitle('Temperature management')
    ax = fig.add_subplot(1,1,1)

    # Set point comfort band
    ax.fill_between(
        range(res_1['env_h']['time_steps']),
        temp_1['indoor_dry_bulb_temperature_set_point'] + temp_1['comfort_band'],
        temp_1['indoor_dry_bulb_temperature_set_point'] - temp_1['comfort_band'],
        color='g',
        alpha=0.1,
        label='Comfort band',
    )

    # Control temperature
    ax.plot(temp_1['indoor_dry_bulb_temperature'], label=algo_names[0] if algo_names else 'RBC', linewidth=2.0)
    ax.plot(temp_2['indoor_dry_bulb_temperature'], label=algo_names[1] if algo_names else 'RL', linewidth=2.0)

    ax.set_ylabel('Temperature (°C)')
    plt.legend()

    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/temperature.png', format='png')
    # plt.show()

def compare_battery(args, res_1, res_2, algo_names=[]):
    # Set figure
    fig = plt.figure(figsize=[20,15])
    fig.suptitle('Electrical storage history')

    # Control 1
    battery_1 = res_1['env_h']['battery']
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title(algo_names[0] if algo_names else 'Rule-Based Control')
    # Charge rate
    ax1.bar(range(res_1['env_h']['time_steps']-1), battery_1['discharge'], color='xkcd:soft blue')
    ax1.set_ylabel('(Dis)Charge (kW/h)')
    ax1.yaxis.label.set_color('xkcd:soft blue')
    # State of charge
    ax2 = ax1.twinx()
    ax2.plot(battery_1['soc'], linewidth=2.0, c='xkcd:orange')
    ax2.set_ylabel('SoC (%)')
    ax2.set_ylim(ymin=-0.05, ymax=1.05)
    ax2.yaxis.label.set_color('xkcd:orange')

    # Control 2
    battery_2 = res_2['env_h']['battery']
    ax3 = fig.add_subplot(2,1,2)
    ax3.set_title(algo_names[1] if algo_names else 'Reinforcement Learning Control')
    # Charge rate
    ax3.bar(range(res_2['env_h']['time_steps']-1), battery_2['discharge'], color='xkcd:soft blue')
    ax3.set_ylabel('(Dis)Charge (kW/h)')
    ax3.yaxis.label.set_color('xkcd:soft blue')
    # State of charge
    ax4 = ax3.twinx()
    ax4.plot(battery_2['soc'], linewidth=2.0, c='xkcd:orange')
    ax4.set_ylabel('SoC (%)')
    ax4.set_ylim(ymin=-0.05, ymax=1.05)
    ax4.yaxis.label.set_color('xkcd:orange')

    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/battery.png', format='png')
    # plt.show()


def evaluate(args, agent_type: str, schema: dict, seed: int=None):
    # Create CityLearn environment
    env = CityLearnEnv(
        schema=schema, 
        central_agent=True,
    )
    # Agent
    if agent_type == 'comfort_rbc':
        agent = ComfortRBC(env)
    elif agent_type == 'advanced_rbc':
        agent = AdvancedRBC(env)
    elif agent_type == 'custom_rbc':
        agent = CustomRBC(env)
    else:
        raise RuntimeError(f'Unknown agent type {agent_type}. Must be either `rbc` or `rl`.')
    
    # Episodic return
    results = {}
    ep_reward = 0.0

    # Step through the environment
    obs, _ = env.reset(seed=args.seed)
    while not env.terminated:
        action = agent.predict(obs)        
        obs, reward, _, _, _ = env.step(action)
        ep_reward += reward[0]

    # Get KPIs
    kpis = get_kpis(env=env)

    # Console log
    print(
        f"{'*'*30}\n CONTROL RESULTS ({agent_type}{f' | seed={seed}' if seed is not None else ''})" +
        f'\n- Reward: {ep_reward}'
    )

    for kpi, value in kpis.items():
        print(f'- {kpi}: {value:.2f}')

    print(f"{'*'*30}")

    # Populate results dict
    results['kpis'] = kpis
    results['env_h'] = {
        'time_steps': env.time_steps,
        'temperature': {
            'indoor_dry_bulb_temperature': env.buildings[0].indoor_dry_bulb_temperature,
            'indoor_dry_bulb_temperature_set_point': env.buildings[0].indoor_dry_bulb_temperature_cooling_set_point,
            'comfort_band': env.buildings[0].comfort_band
        },
        'battery': {
            'soc': env.buildings[0].electrical_storage.soc[:-1],
            'discharge': env.buildings[0].electrical_storage.energy_balance[:-1]
        },
        'net_electricity_consumption': env.buildings[0].net_electricity_consumption
    }

    return results


if __name__ == '__main__':
    config = EvalConfig()
    args = config.args 

    os.makedirs(args.exp_dir, exist_ok=True)

    # Get schema from CityLearn dataset
    schema_obj = CityLearnSchema()
    schema_obj.load(dataset=args.data)

    # Modify schema for testing on a different building
    schema_obj.set_active(key='buildings', items=[f'Building_{args.building}'])

    # Evaluate Rule-based Control agent
    res_comfort_rbc = evaluate(args, 'comfort_rbc', schema_obj.schema)
    res_advanced_rbc = evaluate(args, 'advanced_rbc', schema_obj.schema)

    # Compare results
    compare_temperature(args, res_comfort_rbc, res_advanced_rbc, algo_names=['Comfort RBC', 'Advanced RBC'])
    compare_battery(args, res_comfort_rbc, res_advanced_rbc, algo_names=['Comfort RBC', 'Advanced RBC'])
