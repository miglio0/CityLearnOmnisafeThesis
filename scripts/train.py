# Core
import wandb
import omnisafe
from stable_baselines3 import PPO, SAC

# CityLearn utils
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.agents.rbc import OptimizedRBC as RBCAgent

# Utils
import os
import json
import torch
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from datetime import datetime
from utils import *
from config import TrainConfig


CUSTOM_REWARD_FN = {
    'challenge_reward': {
        'type': 'citylearn.reward_function.Challenge2023Reward',
        'attributes': {
            'band': 2.0,
            'lower_exponent': 2.0,
            'higher_exponent': 2.0,
            'weights': {
                    'w1': 0.3,
                    'w2': 0.1,
                    'w3': 0.6,
                    'w4': 0.0,
                }
        }
    },
    'battery_reward': {
        'type': 'citylearn.reward_function.BatteryAwareComfortReward',
        'attributes': {
            'penalty': 3.0,
            'band': 2.0,
            'lower_exponent': 2.0,
            'higher_exponent': 2.0
        }
    }
}


def main_sb3(args, schema_obj: CityLearnSchema):
    # Create CityLearn environment
    env = CityLearnEnv(
        schema=schema_obj.schema, 
        central_agent=True,
        render_mode='during',
        render_directory=f'{os.path.dirname(__file__)}/render',
        render_session_name=f"{args.algo}_{args.tag}_seed{args.seed}_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}",
    )    

    # Wrap the environment
    sb3_env = NormalizedObservationWrapper(env)
    sb3_env = CityLearnKPIWrapper(sb3_env)

    callback = CityLearnWandbCallback(online=args.wandb)

    if args.wandb:
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            tags=[args.tag, args.wrapper, args.algo],
            name=f"{args.algo}_SB3_seed{args.seed}_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}",
            sync_tensorboard=True
        )

    # SB3 agent
    algo_steps = (env.time_steps-1)

    if args.algo == 'PPO':
        agent = PPO(
            policy='MlpPolicy', 
            env=sb3_env,
            n_steps=algo_steps,
            batch_size=64,      #
            n_epochs=40,        #
            gamma=0.99,         #
            gae_lambda=0.95,    #
            clip_range=0.2,     # -> from omnisafe/omnisafe/configs/on-policy/PPO.yaml
            ent_coef=0.0,       #
            vf_coef=0.001,      #
            max_grad_norm=40.0, #
            target_kl=0.02,     #
            seed=args.seed
        )
    elif args.algo == 'SAC':
        agent = SAC(
            policy='MlpPolicy',
            env=sb3_env,
            buffer_size=1_000_000,  #
            learning_starts=10000,  #
            batch_size=256,         #
            tau=0.005,              # -> from omnisafe/omnisafe/configs/off-policy/SAC.yaml
            gamma=0.99,             # 
            ent_coef=0.2,           #
            seed=args.seed
        )
    else:
        raise NotImplementedError(f'\n{args.algo} algorithm has not been implemented.')
    
    # Agent training
    _ = agent.learn(
        total_timesteps=args.episodes*algo_steps,
        callback=callback
    )

    if args.wandb:
        run.finish()


def main_omnisafe(conf: TrainConfig, schema_obj: CityLearnSchema): 
    # Experiment
    args = conf.args
    seed_dir = f'{exp_dir}/seed{args.seed}'
    os.makedirs(seed_dir, exist_ok=True)

    # RL training steps
    assert args.episodes > 0, f'Invalid number of training episodes: {args.episodes}.'
    ep_steps = schema_obj.schema['simulation_end_time_step'] - schema_obj.schema['simulation_start_time_step']
    train_steps = args.episodes*ep_steps

    # Omnisafe configurations
    custom_cfgs = {
        'seed': args.seed,
        'train_cfgs': {
            'total_steps': train_steps,
            'device': args.device
        },
        'algo_cfgs': {
            'steps_per_epoch': ep_steps,
            'obs_normalize': False # <- DO NOT SET THIS TO True!
        },
        'model_cfgs': {
            'actor_type': 'gaussian_sac',
        },
        'logger_cfgs':
        {
            'use_wandb': args.wandb,
            'wandb_project': args.project,
            'tag': args.tag,
            'entity': args.entity,
            'mode': 'online',
        },
       
        # --- CITYLEARN KEY ARGUMENTS ---
        'env_cfgs': { 
            'schema': schema_obj.schema,
            'central_agent': True,
        }
    }

    if args.render:
        custom_cfgs['env_cfgs'].update({
            'render_mode': 'during',
            'render_directory': f'{os.path.dirname(__file__)}/{exp_dir}/render',
            'render_session_name': f'{args.algo}_{args.tag}_seed{args.seed}',
        })
    
    # Define agent to train on the environment
    agent = omnisafe.Agent(args.algo, 'CityLearn-v0', custom_cfgs=custom_cfgs)
    # Train the agent
    agent.learn()

    # Save policy net
    actor = torch.jit.script(agent.agent._actor_critic.actor.net.cpu())
    actor.save(f'{seed_dir}/actor_net.pt')


if __name__ == '__main__':
    # Configurations
    conf = TrainConfig()
    args = conf.args

    # Experiment
    exp_name = args.name if args.name is not None else datetime.now().strftime('%d-%m-%y_%H:%M')
    exp_dir = f'./experiments/{args.algo}_{exp_name}'
    seed_dir = f'{exp_dir}/seed{args.seed}'
    os.makedirs(exp_dir, exist_ok=True)

    # Get schema from CityLearn dataset
    schema_obj = CityLearnSchema()
    schema_obj.load(dataset=args.data, custom=args.custom)

    # Reward function
    reward_fn = CUSTOM_REWARD_FN.get(args.tag, None)
    if reward_fn is not None:
        schema_obj.set(key='reward_function', value=reward_fn)

    # Modify schema for training 
    train_schema, test_schema = schema_obj.train_test_split(frac=args.frac, mode='train')
    train_schema_obj = CityLearnSchema(schema=train_schema)
    test_schema_obj = CityLearnSchema(schema=test_schema)

    # Train the agent
    main_omnisafe(conf, train_schema_obj)

    # Save configurations
    conf.save_yaml(dir=exp_dir)
    # Save schemas
    schema_dir = f'{exp_dir}/schemas'
    os.makedirs(schema_dir, exist_ok=True)
    schema_obj.save(schema_dir)
    train_schema_obj.save(schema_dir, prefix='train')
    test_schema_obj.save(schema_dir, prefix='eval')