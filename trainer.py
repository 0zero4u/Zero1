

import random
import math
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from ..processor import create_feature_dataframe
from .config import SETTINGS
# IMPORTANT: The model is now the updated one with the Dueling LSTM head
from .tins import MultiTimeframeHybridTIN
from .engine import HierarchicalTradingEnvironment as TradingEnvironment # Renamed for clarity

# The Transition now stores a sequence for 'state' and 'next_state'
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# --- NEW HYPERPARAMETER ---
# Defines how many past time steps the LSTM will look at for each decision.
SEQUENCE_LENGTH = 10 

class ReplayMemory:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

steps_done = 0

def select_action(state_sequence_dict, policy_net):
    """Selects an action based on a sequence of states."""
    global steps_done
    sample = random.random()
    eps_threshold = SETTINGS.training.EPS_END + (SETTINGS.training.EPS_START - SETTINGS.training.EPS_END) * math.exp(-1. * steps_done / SETTINGS.training.EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # The input is already a sequence. Add a batch dimension of 1.
            batched_state = {key: val.unsqueeze(0) for key, val in state_sequence_dict.items()}
            # policy_net will process the sequence and return Q-values for the last step
            return policy_net(batched_state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(SETTINGS.strategy.ACTION_SPACE_SIZE)]], device=SETTINGS.DEVICE, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer):
    train_cfg = SETTINGS.training
    if len(memory) < train_cfg.BATCH_SIZE:
        return
        
    transitions = memory.sample(train_cfg.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=SETTINGS.DEVICE, dtype=torch.bool)
    
    # Collate the batch of state sequences.
    # batch.state is a tuple of dictionaries, where each dictionary contains tensors of shape (Seq, Feat).
    # We stack them to create a batch dictionary with tensors of shape (Batch, Seq, Feat).
    state_batch_dict = {
        key: torch.stack([s[key] for s in batch.state]) for key in batch.state[0].keys()
    }

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch_dict).gather(1, action_batch)

    next_state_values = torch.zeros(train_cfg.BATCH_SIZE, device=SETTINGS.DEVICE)
    non_final_next_states_list = [s for s in batch.next_state if s is not None]

    if non_final_next_states_list:
        # Collate the batch of next_state sequences.
        non_final_next_states_dict = {
            key: torch.stack([s[key] for s in non_final_next_states_list]) for key in non_final_next_states_list[0].keys()
        }
        with torch.no_grad():
            # --- Double DQN Implementation ---
            # 1. Select the best action for the next state using the *policy network*.
            best_actions_next = policy_net(non_final_next_states_dict).argmax(1).unsqueeze(1)
            # 2. Evaluate the Q-value of that chosen action using the *target network*.
            # This decouples selection from evaluation, reducing overestimation.
            q_values_from_target = target_net(non_final_next_states_dict)
            next_state_values[non_final_mask] = q_values_from_target.gather(1, best_actions_next).squeeze(1)
    
    expected_state_action_values = (next_state_values * train_cfg.GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train_model():
    cfg = SETTINGS; train_cfg = cfg.training
    print(f"--- Starting Hybrid Monolithic TIN Model Training on {cfg.DEVICE} (DDQN + Dueling Head) ---")
    print(f"Using sequence length: {SEQUENCE_LENGTH}")
    
    feature_df = create_feature_dataframe("in_sample")
    env = TradingEnvironment(feature_df)
    
    policy_net = MultiTimeframeHybridTIN().to(cfg.DEVICE)
    target_net = MultiTimeframeHybridTIN().to(cfg.DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=train_cfg.LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(train_cfg.MEMORY_SIZE)
    
    for i_episode in range(train_cfg.NUM_EPISODES):
        state = env.reset()
        # Initialize a deque to hold the sequence of states. Prime it with the initial state.
        state_history = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
        
        pbar = tqdm(total=env.max_step - env.current_step, desc=f"Episode {i_episode+1}/{train_cfg.NUM_EPISODES}")
        done = False
        while not done:
            # Assemble the current state sequence from the history deque
            current_state_sequence = {
                key: torch.stack([s[key] for s in state_history]) for key in state.keys()
            }
            
            action = select_action(current_state_sequence, policy_net)
            next_state, reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=cfg.DEVICE)

            # The 'state' for the memory is the sequence that led to the action
            state_for_memory = current_state_sequence
            
            # The 'next_state' for memory is the sequence after the action is taken
            if not done:
                state_history.append(next_state)
                next_state_for_memory = {
                    key: torch.stack([s[key] for s in state_history]) for key in next_state.keys()
                }
            else:
                next_state_for_memory = None

            memory.push(state_for_memory, action, next_state_for_memory, reward)
            state = next_state
            
            optimize_model(memory, policy_net, target_net, optimizer)
            
            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*train_cfg.TAU + target_net_state_dict[key]*(1-train_cfg.TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            pbar.update(1)
        pbar.close()
        
    print("\n🎉 Training complete. 🎉")
    torch.save(policy_net.state_dict(), cfg.get_model_path())
    print(f"Model saved to: {cfg.get_model_path()}")
