# Our agents
import pickle

import pandas as pd
from citylearn.citylearn import EvaluationCondition
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
import seaborn as sns

def plot_temperature(args, results, suffix=''):
    """
    Plot temperature profiles including indoor temperature, outdoor temperature, 
    and comfort band visualization.
    This function creates a comprehensive temperature analysis plot showing:
    - Indoor temperature over time
    - Outdoor temperature over time
    - Comfort band (setpoint ± tolerance range)
    Parameters
    ----------
    results : dict
        Dictionary containing simulation results with nested structure:
        results['env_h']['temperature'] should contain:
        - 'indoor_temperature' : array-like
            Indoor temperature values
        - 'indoor_temperature_set_point' : array-like
            Target setpoint temperature values
        - 'outdoor_temperature' : array-like
            Outdoor temperature values
        - 'comfort_band' : float
            Temperature tolerance band around setpoint (±value)
    suffix : str, optional
        Suffix to append to the output filename (default: '')
    Returns
    -------
    None
        Saves the plot as 'temperature_profile_{suffix}.png'
    Notes
    -----
    - Uses seaborn styling with "whitegrid" style and "talk" context
    - Uses colorblind-friendly palette
    - Saves output at 300 dpi with tight bounding box
    - Figure size: 30x10 inches
    """

    temp_1 = results['env_h']['temperature']
    indoor_temps = temp_1['indoor_dry_bulb_temperature']
    indoor_setpoints = temp_1['indoor_dry_bulb_temperature_set_point']
    comfort_band = temp_1['comfort_band']

    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 5)

    # Ensure arrays match in length
    n = len(indoor_temps)

    # --- Create figure and subplots ---
    fig, axs = plt.subplots(1, 1, figsize=(30, 10))
    # 0️⃣ Temperature profiles
    axs.plot(range(n), indoor_temps, label='Indoor Temperature', color=palette[0], lw=2)
    # Set point comfort band
    axs.fill_between(
        range(n),
        indoor_setpoints + comfort_band,
        indoor_setpoints - comfort_band,
        color='g',
        alpha=0.1,
        label='Comfort band',
    )
    # axs.plot(range(n), outdoor_temps, label='Outdoor Temperature', color=palette[1], lw=2)
    axs.set_ylabel('Temperature [°C]')
    axs.set_title('Temperature Profiles')
    axs.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    plt.tight_layout()
    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    plt.savefig(f'{args.exp_dir}/{mode}_figs/temperature_profile_{suffix}.png', dpi=300, bbox_inches='tight')

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
    # plt.show()

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
    # plt.show()

def plot_energy(
    args,
    res,
    suffix=''
):
    """
    Generate and save a comprehensive energy management visualization with four subplots.
    This function creates a detailed energy profile analysis plot showing building consumption
    components, demand vs. generation, battery control signals, and state of charge over time.
    Parameters
    ----------
    res : dict
        A nested dictionary containing environment and device data with the following structure:
        - res['env_h']['cooling_device']['consumption'] : array-like
            Cooling device power consumption [kW]
        - res['env_h']['dhw']['consumption'] : array-like
            Domestic hot water device power consumption [kW]
        - res['env_h']['non_shiftable_load'] : array-like
            Non-shiftable building load [kW]
        - res['env_h']['battery']['consumption'] : array-like
            Battery charging power [kW]
        - res['env_h']['solar_generation'] : array-like
            Photovoltaic generation [kW]
        - res['env_h']['battery']['discharge'] : array-like
            Battery discharge/charge control signal [kW/h]
        - res['env_h']['battery']['soc'] : array-like
            Battery state of charge [%]
        - res['env_h']['net_electricity_consumption'] : array-like
            Net electricity consumption from grid [kW]
    suffix : str, optional
        Suffix appended to the output filename (default is empty string '')
    Returns
    -------
    None
        The function saves the generated plot to disk as 'energy_profile_{suffix}.png'
    Notes
    -----
    The function generates four stacked/line plots:
    1. Building consumption components (stacked area) with total demand overlay
    2. Building demand, PV generation, and net load comparison
    3. Battery (dis)charge control signal
    4. Battery state of charge over time
    The plot includes an explanatory annotation clarifying net load interpretation
    (positive = grid import, negative = grid export).
    """
    print("*"*50+'\n')
    cooling_device_consumption=res['env_h']['cooling_device']['consumption']
    dhw_device_consumption=res['env_h']['dhw']['consumption']
    non_shiftable_load=res['env_h']['non_shiftable_load']
    battery_charge=res['env_h']['battery']['consumption']
    pv_generation=res['env_h']['solar_generation']
    battery_action=res['env_h']['battery']['discharge']
    battery_soc=res['env_h']['battery']['soc']
    net_load=res['env_h']['net_electricity_consumption']

    time = range(len(cooling_device_consumption))

    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 6)

    # --- Prepare data ---
    pv_generation = -1 * pv_generation  # Flip sign for plotting

    # --- Derived quantities ---
    building_demand = (
        cooling_device_consumption
        + dhw_device_consumption
        + non_shiftable_load
        + battery_charge
    )

    battery_power = battery_action
    label_action = "(Dis)Charge [kW/h]"

    # --- Figure setup ---
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    plt.subplots_adjust(hspace=0.35, right=0.85)

    # 0️⃣ Building Consumption (stacked)
    components = np.vstack([
        cooling_device_consumption,
        dhw_device_consumption,
        non_shiftable_load,
        battery_charge
    ])
    labels = ['Cooling', 'DHW', 'Non-shiftable', 'Battery (Charging)']
    colors = palette[:len(labels)]

    axs[0].stackplot(time, components, labels=labels, colors=colors, alpha=0.9)
    axs[0].plot(time, building_demand, color='black', lw=2, label='Total')
    axs[0].set_ylabel('Power [kW]')
    axs[0].set_title('Building Electricity Consumption Components')

    # Legend on the side (slightly higher)
    legend = axs[0].legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    # 1️⃣ Building Demand, PV, Net Load
    axs[1].plot(time, building_demand, label='Building Demand', color='gray', lw=1.8)
    axs[1].fill_between(time, 0, pv_generation, color=palette[2], alpha=0.3, label='PV Generation')
    axs[1].plot(time, net_load, label='Net Load', color='black', lw=2)
    axs[1].set_ylabel('Power [kW]')
    axs[1].set_title('Building Demand, PV Generation, and Net Load')
    axs[1].legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    # 2️⃣ Battery Action / Power
    axs[2].axhline(0, color='black', lw=0.8)
    sns.lineplot(x=time, y=battery_power, ax=axs[2], color=palette[0], lw=1.8)
    axs[2].set_ylabel(label_action)
    axs[2].set_title('Battery Control Signal (Action)')
    axs[2].set_ylim(-1.1 * np.max(np.abs(battery_power)), 1.1 * np.max(np.abs(battery_power)))

    # 3️⃣ Battery SoC
    sns.lineplot(x=time, y=battery_soc, ax=axs[3], color=palette[4], lw=2)
    axs[3].set_ylabel('State of Charge [%]')
    axs[3].set_xlabel('Time')
    axs[3].set_title('Battery State of Charge (SoC)')

    # Add explanatory note below the first subplot (figure-level annotation)
    fig.text(
        0.80, 0.62,  # position relative to the figure (x, y)
        "Net Load meaning:\n"
        "   • Net Load > 0 → Import from grid\n"
        "   • Net Load < 0 → Export to grid",
        ha='left',
        va='top',
        fontsize=11,
        bbox=dict(
            facecolor='white',
            alpha=0.9,
            edgecolor='gray',
            boxstyle='round,pad=0.4'
        )
    )

    plt.tight_layout()  # leave extra space on right
    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/energy_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

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
        ax.boxplot(kpis_rl, label='RL')
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
    # plt.show()

def compare_kpis_bar(args, res_1, res_2, algo_names=[]):
    """
    Compare Key Performance Indicators (KPIs) between two results and generate a visualization.
    This function creates a horizontal bar chart comparing KPIs from two different 
    results or algorithm runs side by side. The chart is saved as a PNG file in 
    the current working directory.
    Parameters
    ----------
    res_1 : dict
        Dictionary containing results from the first run/algorithm. Must have a 'kpis' 
        key with a dictionary of KPI names and their values.
    res_2 : dict
        Dictionary containing results from the second run/algorithm. Must have a 'kpis' 
        key with a dictionary of KPI names and their values.
    algo_names : list, optional
        List of two strings representing the names of the algorithms/runs being compared. 
        Default is an empty list. If provided, should contain exactly two names for 
        proper labeling in the title and legend.
    Returns
    -------
    None
        The function generates a plot and saves it to disk but does not return a value.
    Notes
    -----
    - The plot is saved as 'kpi_comparison.png' in the current working directory.
    - Uses seaborn for styling with 'whitegrid' style and 'talk' context.
    - Color palette is set to 'colorblind' for better accessibility.
    - Figure size is set to (12, 6) inches at 300 DPI.
    Raises
    ------
    IndexError
        If algo_names list does not have at least 2 elements when accessed.
    KeyError
        If 'kpis' key is not present in res_1 or res_2 dictionaries.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 5)

    kpis_1 = res_1['kpis']
    kpis_2 = res_2['kpis']

    # Create a DataFrame for the KPIs
    kpi_names = list(kpis_1.keys())
    values_1 = [kpis_1[kpi] for kpi in kpi_names]
    values_2 = [kpis_2[kpi] for kpi in kpi_names]

    kpi_df = pd.DataFrame({
        'KPI': kpi_names,
        'Res 1': values_1,
        'Res 2': values_2
    })

    # Set up the horizontal bar plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(kpi_names))

    # Create horizontal bars for both results
    bar1 = plt.barh(index, kpi_df['Res 1'], bar_width, label=algo_names[0], color=palette[0])
    bar2 = plt.barh(index + bar_width, kpi_df['Res 2'], bar_width, label=algo_names[1], color=palette[1])

    # Add labels and title
    plt.ylabel('KPIs')
    plt.xlabel('Values')
    plt.title(f'Comparison of KPIs between {algo_names[0]} and {algo_names[1]}')
    plt.yticks(index + bar_width / 2, kpi_names)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    mode = 'test' if args.test else 'eval' 
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    plt.savefig(f'{args.exp_dir}/{mode}_figs/kpis_bar.png', dpi=300, bbox_inches='tight')

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
    kpis, all_kpis = get_kpis(env=env)

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
    results['all_kpis'] = all_kpis
    results['env_h'] = {
        'time_steps': env.time_steps,
        'temperature': {
            'indoor_dry_bulb_temperature': env.buildings[0].indoor_dry_bulb_temperature,
            'indoor_dry_bulb_temperature_set_point': env.buildings[0].indoor_dry_bulb_temperature_cooling_set_point,
            'outdoor_temperature': env.buildings[0].weather.outdoor_dry_bulb_temperature ,
            'comfort_band': env.buildings[0].comfort_band
        },
        'battery': {
            'soc': env.buildings[0].electrical_storage.soc[:-1],
            'discharge': env.buildings[0].electrical_storage.energy_balance[:-1],
            'consumption': env.buildings[0].electrical_storage.electricity_consumption[:-1]
        },
        'dhw': {
            'soc': env.buildings[0].dhw_storage.soc[:-1],
            'demand': env.buildings[0].dhw_demand[:-1],
            'consumption': env.buildings[0].dhw_electricity_consumption[:-1],
            'energy_from_dhw_storage': env.buildings[0].energy_from_dhw_storage[:-1],
            'energy_from_dhw_device': env.buildings[0].energy_from_dhw_device[:-1]
        },
        'cooling_device': {
            'consumption': env.buildings[0].cooling_device.electricity_consumption[:-1]
        },
        'net_electricity_consumption': env.buildings[0].net_electricity_consumption[:-1],
        'solar_generation': env.buildings[0].solar_generation[:-1],
        'non_shiftable_load': env.buildings[0].non_shiftable_load[:-1],
        'electricity_pricing': env.buildings[0].pricing.electricity_pricing[:-1],
    }

    return results

CHALLENGE_WEIGHTS_PHASE_CUSTOM = {
    'comfort': 0.3,
    'emissions': 0.4,
    'grid_control': 0.3,
    'resilience': 0.0
}

def evaluate_citylearn_challenge(res, weights: dict[str, float]) -> dict[str, float]:
    """
    Evaluates the performance of an agent in the CityLearn Challenge environment using a set of key performance indicators (KPIs)
    and computes a weighted average score based on provided weights.
    Args:
        env (CityLearnEnv): The CityLearn environment instance to evaluate. Must provide an `evaluate` method and support unwrapping.
        weights (dict[str, float]): A dictionary specifying the weights for each high-level score category. 
            Expected keys are 'comfort', 'emissions', 'grid_control', and 'resilience'.
    Returns:
        dict[str, float]: A dictionary containing the evaluation results for each KPI, the computed scores for each category,
            and the final weighted average score. Each entry includes the display name, weight, and value.
    """
    evaluation = {
            'carbon_emissions_total': {'display_name': 'Carbon emissions', 'weight': 0.10},
            'discomfort_proportion': {'display_name': 'Unmet hours', 'weight': 0.30},
            'ramping_average': {'display_name': 'Ramping', 'weight': 0.075},
            'daily_one_minus_load_factor_average': {'display_name': 'Load factor', 'weight': 0.075},
            'daily_peak_average': {'display_name': 'Daily peak', 'weight': 0.075},
            'all_time_peak_average': {'display_name': 'All-time peak', 'weight': 0.075},
            'one_minus_thermal_resilience_proportion': {'display_name': 'Thermal resilience', 'weight': 0.15},
            'power_outage_normalized_unserved_energy_total': {'display_name': 'Unserved energy', 'weight': 0.15},
    }
    data = res['all_kpis']

    data = data[data['level']=='district'].set_index('cost_function').to_dict('index')
    evaluation = {k: {**v, 'value': data[k]['value']} for k, v in evaluation.items()}

    score_comfort = evaluation['discomfort_proportion']['value']
    score_emissions = evaluation['carbon_emissions_total']['value']
    score_grid_control = (
        evaluation['ramping_average']['value'] +
        evaluation['daily_one_minus_load_factor_average']['value'] +
        evaluation['daily_peak_average']['value'] +
        evaluation['all_time_peak_average']['value']
    ) / 4.0
    score_resilience = (
        evaluation['one_minus_thermal_resilience_proportion']['value'] +
        evaluation['power_outage_normalized_unserved_energy_total']['value']
    ) / 2.0

    evaluation['score_comfort'] = {
        'display_name': 'Comfort score',
        'weight': weights['comfort'],
        'value': score_comfort
    }
    evaluation['score_emissions'] = {
        'display_name': 'Emissions score',
        'weight': weights['emissions'],
        'value': score_emissions
    }
    evaluation['score_grid_control'] = {
        'display_name': 'Grid control score',
        'weight': weights['grid_control'],
        'value': score_grid_control
    }
    evaluation['score_resilience'] = {
        'display_name': 'Resilience score',
        'weight': weights['resilience'],
        'value': score_resilience
    }
    weighted_resilience = weights['resilience'] * score_resilience
    evaluation['average_score'] = {
        'display_name': 'Score',
        'weight': None,
        'value': (
            weights['comfort'] * score_comfort +
            weights['emissions'] * score_emissions +
            weights['grid_control'] * score_grid_control +
            (weighted_resilience if not np.isnan(weighted_resilience) else 0.0)
        )
    }

    return evaluation

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
    with open(os.path.join(f'{args.exp_dir}', f"rbc_res.pkl"), 'wb') as f:
        pickle.dump(res_rbc, f)

    # Evaluate all the seeds of the RL experiment
    res_rl = defaultdict(dict)
    for i in range(1, len(glob(f'{args.exp_dir}/seed*'))+1):
        res = evaluate(args, 'rl', schema_obj.schema, seed=i)
        res_rl[f'seed{i}'] = res
        with open(os.path.join(f'{args.exp_dir}', f"rl_res.pkl"), 'wb') as f:
            pickle.dump(res, f)

    # Compare results
    compare_kpis(args, res_rbc, res_rl)
    # compare_battery(args, res_rbc, res_rl)
    compare_temperature(args, res_rbc, res_rl)
    plot_temperature(args, res_rbc, suffix='rbc')
    plot_temperature(args, res_rl['seed1'], suffix='rl')
    plot_energy(args, res_rbc, suffix='rbc')
    plot_energy(args, res_rl['seed1'], suffix='rl')
    compare_kpis_bar(args, res_rbc, res_rl['seed1'], algo_names=['RBC', 'RL'])


    scores_rl = evaluate_citylearn_challenge(
        res=res_rl['seed1'],
        weights = CHALLENGE_WEIGHTS_PHASE_CUSTOM
    )
    scores_rl_df = pd.DataFrame.from_dict(scores_rl, orient='index')
    scores_rl_df.to_csv(os.path.join(f'{args.exp_dir}', f"rl_scores.csv"))
    
    scores_rbc = evaluate_citylearn_challenge(
        res=res_rbc,
        weights = CHALLENGE_WEIGHTS_PHASE_CUSTOM
    )
    scores_rbc_df = pd.DataFrame.from_dict(scores_rbc, orient='index')
    scores_rbc_df.to_csv(os.path.join(f'{args.exp_dir}', f"rbc_scores.csv"))