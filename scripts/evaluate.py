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


def compare_temperature(args, res_rbc, res_rl):
    # RBC temperature
    temp_rbc = res_rbc['env_h']['temperature']

    # RL temperature stats
    temp_rl = np.vstack([res_rl[seed]['env_h']['temperature']['indoor_dry_bulb_temperature'] for seed in res_rl.keys()])
    avg_temp = np.mean(temp_rl, axis=0)
    std_temp = np.std(temp_rl, axis=0)
    
    fig = plt.figure(figsize=[30,10])
    fig.suptitle('Temperature management')
    ax = fig.add_subplot(1,1,1)

    # Set point comfort band
    ax.fill_between(
        range(res_rbc['env_h']['time_steps']),
        temp_rbc['indoor_dry_bulb_temperature_set_point'] + temp_rbc['comfort_band'],
        temp_rbc['indoor_dry_bulb_temperature_set_point'] - temp_rbc['comfort_band'],
        color='g',
        alpha=0.1,
        label='Comfort band',
    )

    # Control temperature
    ax.plot(temp_rbc['indoor_dry_bulb_temperature'], label='RBC', linewidth=2.0)
    ax.plot(avg_temp, label='RL', linewidth=2.0)
    ax.fill_between(
        range(res_rbc['env_h']['time_steps']),
        avg_temp + std_temp,
        avg_temp - std_temp,
        color='orange',
        alpha=0.3
    )

    ax.set_ylabel('Temperature (°C)')
    plt.legend()

    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/temperature.png', format='png')
    plt.show()

def compare_battery(args, res_rbc, res_rl):
    # Set figure
    fig = plt.figure(figsize=[20,15])
    fig.suptitle('Electrical storage history')

    # Rule-based Control
    battery_rbc = res_rbc['env_h']['battery']
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('Rule-Based Control')
    # Charge rate
    ax1.bar(range(res_rbc['env_h']['time_steps']-1), battery_rbc['discharge'], color='xkcd:soft blue')
    ax1.set_ylabel('(Dis)Charge (kW/h)')
    ax1.yaxis.label.set_color('xkcd:soft blue')
    # State of charge
    ax2 = ax1.twinx()
    ax2.plot(battery_rbc['soc'], linewidth=2.0, c='xkcd:orange')
    ax2.set_ylabel('SoC (%)')
    ax2.set_ylim(ymin=-0.05, ymax=1.05)
    ax2.yaxis.label.set_color('xkcd:orange')

    # Reinforcement Learning Control
    bx1 = fig.add_subplot(2,1,2)
    bx1.set_title('Reinforcement Learning Control')
    # Charge rate
    discharge_rl = np.vstack([res_rl[seed]['env_h']['battery']['discharge'] for seed in res_rl.keys()])
    avg_discharge = np.mean(discharge_rl, axis=0)
    std_discharge = np.std(discharge_rl, axis=0)
    bx1.bar(range(res_rbc['env_h']['time_steps']-1), avg_discharge, yerr=std_discharge, color='xkcd:soft blue', ecolor='xkcd:brick red')
    bx1.set_ylabel('(Dis)Charge (kW/h)')
    bx1.yaxis.label.set_color('xkcd:soft blue')
    # State of charge
    soc_rl = np.vstack([res_rl[seed]['env_h']['battery']['soc'] for seed in res_rl.keys()])
    avg_soc = np.mean(soc_rl, axis=0)
    std_soc = np.std(soc_rl, axis=0)
    bx2 = bx1.twinx()
    bx2.plot(avg_soc, linewidth=2.0, c='xkcd:orange')
    bx2.fill_between(
        range(res_rbc['env_h']['time_steps']-1),
        avg_soc + std_soc,
        avg_soc - std_soc,
        color='xkcd:orange',
        alpha=0.2
    )
    bx2.set_ylabel('SoC (%)')
    bx2.set_ylim(ymin=-0.05, ymax=1.05)
    bx2.yaxis.label.set_color('xkcd:orange')

    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/battery.png', format='png')
    plt.show()


def compare_kpis(args, res_rbc, res_rl):
    # RBC KPIs
    kpis_rbc = res_rbc['kpis']

    # Set figure
    fig = plt.figure(figsize=[20,15])
    fig.suptitle('Key Performance Indicators (KPIs) comparison')
    handles, labels = [], []

    # Plot KPIs comparison
    for i, kpi in enumerate(kpis_rbc.keys()):
        ax = fig.add_subplot(2, 3, i+1)

        # RL KPIs
        kpis_rl = np.array([res_rl[seed]['kpis'][kpi] for seed in res_rl.keys()])

        # Bar plot
        ax.set_title(kpi)
        ax.boxplot(kpis_rl, orientation='vertical', label='RL')
        ax.axhline(y=kpis_rbc[kpi], linestyle='--', color='red', label='RBC')
        ax.set_ylim(
            ymin=min(kpis_rbc[kpi], kpis_rl.min())-0.02,
            ymax=max(kpis_rbc[kpi], kpis_rl.max())+0.02
        )
        ax.set_xticks([])

        # Collect handles/labels from this subplot
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicate labels (since 'RBC' and 'RL' repeat)
    by_label = OrderedDict(zip(labels, handles))

    # Add one legend for the whole figure
    fig.legend(
        by_label.values(), 
        by_label.keys(),
        loc='upper center', 
        ncol=2, 
        bbox_to_anchor=(0.5, 0.95), 
    )

    mode = 'test' if args.test else 'eval' 
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/kpis.png', format='png')
    plt.show()


def evaluate(args, agent_type: str, schema: dict, seed: int=None):
    # Create CityLearn environment
    env = CityLearnEnv(
        schema=schema, 
        central_agent=True,
    )
    if agent_type == 'rl':
        env = NormalizedObservationWrapper(env)

    # Agent
    if agent_type == 'rbc':
        agent = ComfortRBC(env)
    elif agent_type == 'rl':
        assert seed is not None, f'Missing experiment seed.'
        fname = glob(f'{args.exp_dir}/seed{seed}/*.pt')[0]
        agent = OmnisafeActorWrapper(fname=fname, env=env)
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

    # Read the schema.json file of the experiment
    schema_obj = CityLearnSchema()
    if not args.test:
        with open(f'{args.exp_dir}/schemas/eval_schema.json') as f:
            schema_obj.schema = json.load(f)
    else:
        with open(f'{args.exp_dir}/schemas/base_schema.json') as f:
            schema_obj.schema = json.load(f)

        # Modify schema for testing on a different building
        schema_obj.set_active(key='buildings', items=[f'Building_{args.building}'])

    # Evaluate Rule-based Control agent
    res_rbc = evaluate(args, 'rbc', schema_obj.schema)

    # Evaluate all the seeds of the RL experiment
    res_rl = defaultdict(dict)
    for i in range(1, len(glob(f'{args.exp_dir}/seed*'))+1):
        res_rl[f'seed{i}'] = evaluate(args, 'rl', schema_obj.schema, seed=i)

    # Compare results
    compare_kpis(args, res_rbc, res_rl)
    compare_battery(args, res_rbc, res_rl)
    compare_temperature(args, res_rbc, res_rl)