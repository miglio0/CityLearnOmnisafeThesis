# Core
import wandb
import omnisafe
from stable_baselines3 import PPO, SAC

# CityLearn utils
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

# Utils
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from datetime import datetime
from utils import *


def main_sb3(args):
    # Get schema from CityLearn dataset
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = DataSet().get_schema(args.data)

    # Create CityLearn environment
    env = CityLearnEnv(schema=schema, central_agent=True)

    # Wrap the environment
    sb3_env = NormalizedObservationWrapper(env)
    sb3_env = CityLearnKPIWrapper(sb3_env)

    callback = CityLearnWandbCallback() if args.wandb else None

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


def main_omnisafe(args):
    # Get schema from CityLearn dataset
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data)

    # Omnisafe configurations
    custom_cfgs = {
        'seed': args.seed,
        'train_cfgs': {
            'total_steps': 719000,
        },
        'algo_cfgs': {
            'steps_per_epoch': 719,
            'obs_normalize': False
        },
        'model_cfgs': {
            'actor_type': 'gaussian_sac',
        },
        'logger_cfgs':
        {
            # use wandb for logging
            'use_wandb': args.wandb,
            'wandb_project': args.project,
            'tag': args.tag,
            'entity': args.entity,
            'mode': 'online',
        },
       
        # --- CITYLEARN KEY ARGUMENTS ---
        'env_cfgs': { 
            'schema': schema,
            'central_agent': True,
            # 'reward_function': 'citylearn.reward_function.Challenge2023Reward',
            # 'reward_function_kwargs': {
            #     'band': 2.0,
            #     'lower_exponent': 2.0,
            #     'higher_exponent': 2.0,
            #     'weights': {
            #             'w1': 0.3,
            #             'w2': 0.1,
            #             'w3': 0.6,
            #             'w4': 0.0,
            #         }
            # }
        }
    }
    
    # Define agent to train on the environment
    agent = omnisafe.Agent(args.algo, 'CityLearn-v0', custom_cfgs=custom_cfgs)
    # Train the agent
    agent.learn()


if __name__ == '__main__':
    # Configurations
    conf = Config()
    args = conf.args

    if args.wrapper == 'sb3':
        main_sb3(args)
    else:
        main_omnisafe(args)