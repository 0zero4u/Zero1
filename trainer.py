# rl-main/trainer.py

import random
import math
from collections import namedtuple, deque
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from ..data.processor import create_bars_from_trades # Assuming this is adapted to get bars
from .config import SETTINGS
from .tins import DQN_TIN
from .engine import TradingEnvironment

# Define the structure for a single transition in our environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """A cyclic buffer of bounded size that stores the transitions observed recently."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train_model():
    """Main function to orchestrate the DQN training process."""
    cfg = SETTINGS
    train_cfg = cfg.training
    
    print(f"--- Starting TIN Model Training on {cfg.DEVICE} ---")
    
    # 1. Prepare Data and Environment
    # In a real scenario, you'd get this from your data processing pipeline for the in-sample period.
    # We use a placeholder function here.
    bars_df = create_bars_from_trades("in_sample") # This needs to point to the correct function
    env = TradingEnvironment(bars_df)

    # 2. Initialize Networks, Optimizer, and Memory
    policy_net = DQN_TIN().to(cfg.DEVICE)
    target_net = DQN_TIN().to(cfg.DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target network is not trained directly

    optimizer = optim.AdamW(policy_net.parameters(), lr=train_cfg.LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(train_cfg.MEMORY_SIZE)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = train_cfg.EPS_END + (train_cfg.EPS_START - train_cfg.EPS_END) * \
            math.exp(-1. * steps_done / train_cfg.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(cfg.strategy.ACTION_SPACE_SIZE)]], device=cfg.DEVICE, dtype=torch.long)

    def optimize_model():
        if len(memory) < train_cfg.BATCH_SIZE:
            return
        transitions = memory.sample(train_cfg.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=cfg.DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(train_cfg.BATCH_SIZE, device=cfg.DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * train_cfg.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    # 3. Main Training Loop
    print(f"Running for {train_cfg.NUM_EPISODES} episodes...")
    for i_episode in range(train_cfg.NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        pbar = tqdm(total=len(env.df) - cfg.strategy.LOOKBACK_WINDOW, desc=f"Episode {i_episode+1}/{train_cfg.NUM_EPISODES}")
        while True:
            action = select_action(state)
            next_state, reward, done = env.step(action.item())
            episode_reward += reward
            reward_tensor = torch.tensor([reward], device=cfg.DEVICE)

            memory.push(state, action, next_state, reward_tensor)
            state = next_state
            
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*train_cfg.TAU + target_net_state_dict[key]*(1-train_cfg.TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            pbar.update(1)
            if done:
                pbar.close()
                break
        
        print(f"  -> Episode {i_episode+1} complete. Total Reward: {episode_reward:.2f}")

    # 4. Save Model
    model_path = cfg.get_model_path()
    torch.save(policy_net.state_dict(), model_path)
    print(f"\n✅ Training complete. Model saved to: {model_path}")
