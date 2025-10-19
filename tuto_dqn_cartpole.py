# Source : https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1",render_mode="human")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


# 1 - Replay Memory / Experience Replay
# Note : RM stores the transitions that the agent obseves, allowing us to reuse this data later.
# By sampling from it randomly, the transitions that build up a batch are decorrelated.
# It has been shown that this greatly stabilizes and improves the DQN training procedure.

# 1.1 - Transition
# This class (named tuple) represents a single transition in our environment.
# It maps (state,action) pairs to their (next_state,reward) result, with the state being the screen difference image as describe later on.
Transition = namedtuple('Transition',
                        ('state','action','next_state','reward'))

# 1.2 - ReplayMemory
# A cyclic buffer of bounded size that hold the transitions observed recently
# .sample() method for selecting a random batch of transitions for training
class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
        #file doublement chaine
        #maxlen=capacity : Si la memoire est pleine, les ancienens transitions sont automatiquements supprimées lorsque de nouvelle sont ajoutées
        #capacity : taille du buffer

    def push(self,*args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        # on crée un isntance de Transition() à partir dse arguments fournis, puis on l'ajoute à la mémoire

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
        # selectionne aleatoirement N=batch_size transitions parmi la memoire (casse la corrélation temporelle entre les transitiosn consécutives)
    
    def __len__(self):
        return len(self.memory)
        # permet d'obtenir la taille actuelle du buffer

    
# 2 - Q-Network
# Our model will be a feed forward neural network that takes the difference between the current and previous screen patches.
# It has two outputs, representing Q(s,left) and Q(s,right) (where s in the input to the network)
# In effect, the network is trying to predict the expected return of taking each action given the current input.

class DQN(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(in_features=n_observations,out_features=128)
        self.layer2 = nn.Linear(in_features=128,out_features=128)
        self.layer3 = nn.Linear(in_features=128,out_features=n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...])
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# 3 - Training
# BATCH_SIZE is the number of transitions sampled from the replay buffer
BATCH_SIZE = 128

# GAMMA is the discout factor as mentionned in the previous section
GAMMA = 0.99

# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
EPS_START = 0.9
EPS_END = 0.01

# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 2500

# TAU is the update rate of the target network
TAU = 0.005

# LR is the learning rate of the AdamW optimizer
LR = 3e-4

# Get number of actions from gym action space
n_actions = env.action_space.n #2 pour cartpole

# Get the number of state observations
state, info = env.reset() #remet l'environnement a son etat initial
n_observations = len(state) # cb de valeurs composent un etat = taille de l'entrée du NN

policy_net = DQN(n_observations,n_actions).to(device) #reseau principal : choisit les actions à executer dans l'environnement (mis a jour a chaque pas)
target_net = DQN(n_observations,n_actions).to(device) #evite que le reseau aprenne sur des cibles qui changent trop vite (mis a jour tous les N pas, c.f. TAU)
target_net.load_state_dict(policy_net.state_dict()) # copie des poids et biais du reseau principal dans le reseau cible

optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad=True)
# amsgrad=True ameliore la stabilite de convergence dans certains cas (option de l'algorithme Adam)
memory = ReplayMemory(10000)

steps_done = 0 
#pour compter le nombre d'actions déjà effectuées par l'agent
# utilisé pour faire décroître progressivement la proba d'exploration epsilon (epsilon-greedy)

# select_action() will select ana ction according to an epsilon greedy policy
# We'll sometimes use our model for choosing the action, and sometimes, we'll just sample one uniformly.
# The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END
# EPS_DECAY controls the rate of the decay
def select_action(state):
    global steps_done
    #On tire aleatoirement un float entre 0 et 1
    sample = random.random()

    #calcul du seuil epsilon : formule du décroissement exponentiel de epsilon
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    # plus le temps passe plus epsilon tend vers EPS_END
    # moins de hasard avec le temps -> l'agent devient plus intelligent/confiant dans sa politique de choix d'action
    
    steps_done += 1

    if sample > eps_threshold:
        #Exploitation : on choisit l'action avec la plus grande Q-value (strategie greedy)
        with torch.no_grad(): #on desactive le calcul du gradient i.e on ne fait pas d'apprentissage ici
            # t.max(1) will return the largest column value of each row
            # second column on max result is index of where max element was
            # found, so we pick action with the largest expected reward
            
            # policy_net(state) renvoie un vecteur de Q-values pour chaque action
            # .max(1) renvoie la Q-value la plus élevée (la valeur max sur l'axe des actions), et l'indice de cette action (à exécuter)
            return policy_net(state).max(1).indices.view(1,1)
    else:
        #Exploration : l'agent essaie une action au hasard pour découvrir de nouvelles strategies
        return torch.tensor([[env.action_space.sample()]],device=device,dtype=torch.long)
    
episode_durations = []

# plot_durations() : a helper for plotting the duration of episodes, along with an average over the last 100 episodes (the measure used in the official evaluations)
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations,dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())

    plt.pause(0.001) #pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# 4 - Training loop
# optimize_model() performs a single step of the optimizaiton
# it first samples a batch, concatenates all the tensors into a single one
# computes Q(s_t,a_t) and V(s_t+1) = max_aQ(s_t+1,a), and combines them intou our loss
# by definition : V(s)=0 if s i a terminal state.
# We also use a target network to compute V(s_t+1) for added stability
# The target network is updated at every with a soft update controlled by TAU

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch. THis converts batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t,a) - the model computes Q(s_t)
    # Then we select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1,action_batch)

    # Compute V(s_{t+1}) for all next states
    # Expected values of actions for non_final_next_states are computed based on the "older" target_net
    # selecting their best reward with max(1).values
    # THis is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()

# Main Training Loop
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 500
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environemnt and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward],device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation,dtype=torch.float32,device=device).unsqueeze(0)
        
        # Store the transition in memory
        memory.push(state,action,next_state,reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimizaiton (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t+1)
            plot_durations()
            break
    
    if i_episode % 25 == 0:
        env.render()

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
