import numpy as np
import contextlib
import torch

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)
        
    def p(self, next_state, state, action):
        raise NotImplementedError()
    
    def r(self, next_state, state, action):
        raise NotImplementedError()
        
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps
        
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state
        
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()

        
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.size = len(self.lake_flat)
        self.width = int(np.sqrt(self.size))
        
        self.holeStates = []
        for i in range(len(self.lake_flat)):
            if self.lake_flat[i] == '#': self.holeStates.append(i)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        
    actions = {'w':0, 'a':1, 's':2, 'd':3}
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def isAbsorbingState(self, state): return state == self.size
    def isStartState(self, state): return state == np.where(self.lake_flat == '&')[0][0]
    def isGoalState(self, state): return state == np.where(self.lake_flat == '$')[0][0]
    def isHoleState(self, state): return state in self.holeStates
    
    def p(self, next_state, state, action):
        if self.isAbsorbingState(state) or self.isGoalState(state) or self.isHoleState(state):   
            return self.isAbsorbingState(next_state)
        elif self.isAbsorbingState(next_state): return 0
        
        # Describes [y, x] movement of the action
        actions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        a = actions[action]
        w = self.width
        ps = np.ndarray((w, w))
        ps.fill(0)
        getCoords = lambda a: (np.array([state//w, state%w]) + np.array(a)).clip(0, self.width - 1)
        ps[getCoords(a)[0], getCoords(a)[1]] = 1 - self.slip
        for _a in actions: ps[getCoords(_a)[0], getCoords(_a)[1]] += self.slip / self.n_actions
        return ps.flatten()[next_state]
    
    def r(self, _, state, __):
        return 1 if self.isGoalState(state) else 0
    
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))
        
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)
    inner = lambda s, a: sum([env.p(s_, s, a) * (env.r(s_, s, a) + gamma * value[s_]) for s_ in range(env.n_states)])
    outer = lambda s: inner(s, policy[s])
    for _ in range(max_iterations):
        v = value.copy();
        value = np.array([outer(s) for s in range(env.n_states)])
        if max([abs(v[i] - value[i]) for i in range(len(value))]) < theta: break
    return value
    
def policy_improvement(env, value, gamma):
    inner = lambda s, a: sum([env.p(s_, s, a) * (env.r(s_, s, a) + gamma * value[s_]) for s_ in range(env.n_states)])
    outer = lambda s: np.argmax([inner(s, a) for a in range(env.n_actions)])
    return np.array([outer(s) for s in range(env.n_states)])
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    
    old_policy = None
    while not np.array_equal(policy, old_policy):
        old_policy = policy.copy()
        policy = policy_improvement(env, value, gamma)
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    value = np.zeros(env.n_states) if value is None else np.array(value, dtype=np.float)
    # Define math function as outer
    inner = lambda s, a: sum([env.p(s_, s, a) * (env.r(s_, s, a) + gamma * value[s_]) for s_ in range(env.n_states)])
    outer = lambda s: np.max([inner(s, a) for a in range(env.n_actions)])
    # Iterate
    for _ in range(max_iterations):
        _value = value.copy()
        value = np.array([outer(s) for s in range(env.n_states)])
        if max([abs(_value[i] - value[i]) for i in range(len(value))]) < theta: break
    inner = lambda s, a: sum([env.p(s_, s, a) * value[s_] for s_ in range(env.n_states)])
    outer = lambda s: np.argmax([inner(s, a) for a in range(env.n_actions)])
    policy = np.array([outer(s) for s in range(env.n_states)])
    return policy, value

#region Later
def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # Select action a for state s according to an ε-greedy policy based on Q
        qs = lambda s: {a : q[s, a] for a in range(env.n_actions)}
        random_max = lambda s:  (a_maxs := [a for a, q in qs(s).items() if q == max(qs(s).values())])[random_state.randint(0, len(a_maxs))]
        select_a = lambda s: random_max(s) if random_state.rand() > epsilon[i] else random_state.randint(0, env.n_actions)
        a = select_a(s)
        
        while (not env.isAbsorbingState(s)):
            s_, r = env.draw(s, a)
            a_ = select_a(s_)
            q[s, a] += eta[i] * (r + gamma*q[s_, a_] - q[s,a])
            s, a = s_, a_
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # Select action a for state s according to an ε-greedy policy based on Q
        qs = lambda s: {a : q[s, a] for a in range(env.n_actions)}
        random_max = lambda s:  (a_maxs := [a for a, q in qs(s).items() if q == max(qs(s).values())])[random_state.randint(0, len(a_maxs))]
        select_a = lambda s: random_max(s) if random_state.rand() > epsilon[i] else random_state.randint(0, env.n_actions)
        
        while (not env.isAbsorbingState(s)):
            a = select_a(s)
            s_, r = env.draw(s, a)
            q[s, a] += eta[i] * (r + gamma*max(qs(s).values()) - q[s,a])
            s = s_
              
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:

    return theta    

class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env

        lake = self.env.lake

        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])

        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]

        self.state_image = {lake.absorbing_state: 
                            np.stack([np.zeros(lake.shape)] + lake_image)}
        for state in range(lake.size):
            # TODO: 
            return

    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
        
class DeepQNetwork(torch.nn.Module):
    def __init__(self, env, learning_rate, kernel_size, conv_out_channels, 
                 fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.conv_layer = torch.nn.Conv2d(in_channels=env.state_shape[0], 
                                          out_channels=conv_out_channels,
                                          kernel_size=kernel_size, stride=1)

        h = env.state_shape[1] - kernel_size + 1
        w = env.state_shape[2] - kernel_size + 1

        self.fc_layer = torch.nn.Linear(in_features=h * w * conv_out_channels, 
                                        out_features=fc_out_features)
        self.output_layer = torch.nn.Linear(in_features=fc_out_features, 
                                            out_features=env.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        
        # TODO: 

    def train_step(self, transitions, gamma, tdqn):
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])

        q = self(states)
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)

        target = torch.Tensor(rewards) + gamma * next_q

        # TODO: the loss is the mean squared error between `q` and `target`
        loss = 0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    
        
        
class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        # TODO:
        return;
        
        
def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon, 
                            batch_size, target_update_frequency, buffer_size, 
                            kernel_size, conv_out_channels, fc_out_features, seed):
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, 
                       fc_out_features, seed=seed)
    tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, 
                        fc_out_features, seed=seed)

    epsilon = np.linspace(epsilon, 0, max_episodes)

    for i in range(max_episodes):
        state = env.reset()

        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                with torch.no_grad():
                    q = dqn(np.array([state]))[0].numpy()

                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)

            next_state, reward, done = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())

    return dqn
#endregion

def main():
    seed = 0
    
    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]
    
    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
    
    
    
    max_steps = len([cell for row in lake for cell in row])
        
    env = FrozenLake(lake, slip=0.1, max_steps=max_steps, seed=seed)
    gamma = 0.9
    
    # good_policy = np.zeros(env.n_states, dtype=int)
    # good_policy.fill(3)
    # for s in [2, 6, 10]: good_policy[s] = 2
    
    # bad_policy = np.zeros(env.n_states, dtype=int)
    
    # policy, value = value_iteration(env, gamma, 0.0001, 100)
    # print(policy[:-1].reshape(4, 4))
    # print(value[:-1].reshape(4, 4))
    
    # policy, value = policy_iteration(env, gamma, 0.0001, 100, bad_policy)
    # print(policy[:-1].reshape(4, 4))
    # print(value[:-1].reshape(4, 4))
    
    print('# Model-based algorithms')

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 4000

    print('')

    # print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma,
    #                       epsilon=0.5, seed=seed)
    # env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
                               epsilon=0.5, seed=seed)
    env.render(policy, value)

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma, 
                              epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma, 
                                   epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    image_env = FrozenLakeImageWrapper(env)

    print('## Deep Q-network learning')

    dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001, 
                                  gamma=gamma,  epsilon=0.2, batch_size=32,
                                  target_update_frequency=4, buffer_size=256,
                                  kernel_size=3, conv_out_channels=4,
                                  fc_out_features=8, seed=4)
    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)

main()