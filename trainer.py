import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ..processor import create_bars_from_trades
from .config import SETTINGS
from .tins import MultiTimeframeFeatureExtractor
from .engine import HierarchicalTradingEnvironment

def train_model():
    cfg = SETTINGS
    train_cfg = cfg.training
    print(f"--- Starting Hybrid TIN Model Training with SB3 PPO on {cfg.DEVICE} ---")

    # 1. Create the Environment
    bars_df = create_bars_from_trades("in_sample")
    env = HierarchicalTradingEnvironment(bars_df)
    # Optional: Check if the custom environment is compliant
    # check_env(env) 
    # print("✅ Environment check passed.")

    # 2. Define the Custom Policy
    # "MlpPolicy" is used, but its feature extractor will be replaced by our custom one.
    # SB3 will automatically attach Actor and Critic heads to our extracted features.
    policy_kwargs = dict(
        features_extractor_class=MultiTimeframeFeatureExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden_size=64, # Can pass arguments to the extractor here
            lstm_layers=2
        ),
        # PPO's actor-critic network architecture after feature extraction
        net_arch=dict(pi=[128, 64], vf=[128, 64]) 
    )

    # 3. Instantiate the PPO Agent
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=train_cfg.N_STEPS,
        batch_size=train_cfg.BATCH_SIZE,
        n_epochs=train_cfg.N_EPOCHS,
        gamma=train_cfg.GAMMA,
        gae_lambda=train_cfg.GAE_LAMBDA,
        clip_range=train_cfg.CLIP_RANGE,
        ent_coef=train_cfg.ENT_COEF,
        learning_rate=train_cfg.LEARNING_RATE,
        verbose=1,
        device=cfg.DEVICE,
        tensorboard_log=f"{cfg.BASE_PATH}/tensorboard_logs/"
    )

    # 4. Train the Agent
    print("--- Starting PPO Training ---")
    model.learn(total_timesteps=train_cfg.TOTAL_TIMESTEPS, progress_bar=True)
    
    # 5. Save the Trained Model
    model_path = cfg.get_model_path()
    model.save(model_path)
    print("\n🎉 Training complete. 🎉")
    print(f"Model saved to: {model_path}")
