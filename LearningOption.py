import numpy as np
from option import Option

""" Agent planning using Intra-Option Q-learning """

class IntraOptionQLearningAgent():
    def __init__(self,options,gamma=0.9, alpha=0.125):
        
        self.options=options
    
        self.gamma = gamma
        self.alpha = alpha

        self.current_option_idx = None
        # Keep track of action number corresponding to last action taken
        self.last_action_taken = None
        self.rewards = []  # Rewards for current option
        self.trajectory=[] # states visited in a trajectory for current option
        self.pis=[]

        # Initialize option value table, and occurrence counts table
        n_states = len(self.options[0].I)
        n_options = len(self.options)
        self.Q = np.zeros((n_states, n_options))
        self.N = np.zeros((n_states, n_options)) #count of visit

    def _resetCurrentOption(self):
        self.rewards = []
        self.trajectory=[]
        self.pis=[]
        self.current_option_idx = None
        
    def _pickNewOptionEpsilonGreedily(self, state, epsilon):
        # Iterate over options, keeping track of all available options
        # and the index of best option seen so far
        available_options = []
        available_options_idx=[]
        best_option_index = 0 
        s = state
        for i in range(len(self.options)):
            if self.options[i].I[s] == 1:
                available_options.append(self.options[i])
                available_options_idx.append(i)
                if self.Q[s, i] > self.Q[s, best_option_index]:
                    best_option_index = i

        # Pick greedy option with probability (1 - epsilon)       
        # Pick random action with probability epsilon
        if len(available_options_idx)==0:
            print(s)
        
        if np.random.rand() <= epsilon:
            best_option_index=np.random.choice(available_options_idx)

        self.current_option_idx = best_option_index
        self.rewards=[]
        self.pis=[]
        self.trajectory=[state]
    
    
    # Most of the work is done here
    def _updateQValuesIntra(self):
        k=len(self.trajectory)-1
        s1_idx=0
        for s1_idx in range(k):
            s1=self.trajectory[s1_idx]
            s2=self.trajectory[s1_idx+1]
            reward=self.rewards[s1_idx]
            last_action=self.pis[s1_idx]
            self._updateQValuesIntra_core(s1, reward, last_action,s2)

    def _updateQValuesIntra_core(self, state, reward, last_action,next_state):
        
        # List of all options consistent with last action taken
        consistent_option_index = []
        for option_idx in range(len(self.options)):
            if option_idx == self.current_option_idx:
                continue #don't update current option again
            option=self.options[option_idx]
            if option.pi[state] == last_action:
                consistent_option_index.append(option_idx)
        
        # Update table for every option consistent with last action taken
        Q=self.Q.copy()

        for option_idx in consistent_option_index:
            option=self.options[option_idx]
            
            # Update Q table
            U = (1 - option.beta[next_state]) * self.Q[next_state, option_idx] + \
                    option.beta[next_state] * np.max(self.Q[next_state])
            target = reward + self.gamma * U
            
            old_Q=Q[state, option_idx]
            self.Q[state, option_idx] = old_Q + self.alpha * (target - old_Q)
    
    def _updateQValueOption(self):
        o = self.current_option_idx
        s2 = self.trajectory[-1]      
        
        k=len(self.trajectory)-1
        for s1_idx in range(k):
            s1=self.trajectory[s1_idx]
            self.N[s1, o] += 1
            cumulative_reward=np.sum(np.array([self.rewards[s1_idx_]*self.gamma**(s1_idx_-s1_idx) for s1_idx_ in range(s1_idx,k)]))
            #alpha = (1. / self.N[s1, o])
            target = cumulative_reward + \
                (self.gamma ** (k-s1_idx)) * np.max(self.Q[s2])
            self.Q[s1, o] += self.alpha * (target - self.Q[s1, o])

    
    def epsilonGreedyPolicy(self, state, epsilon=0.1):
        if self.current_option_idx is None:
            self._pickNewOptionEpsilonGreedily(state, epsilon)
        current_option=self.options[self.current_option_idx]
        action, action_number = current_option.pickAction(state)
        # Record action number corresponding to last action taken
        self.last_action_taken = action_number
        
        return action,action_number
    
    def epsilonGreedyPolicy_Interruption(self, state, epsilon=0.1):

        self._pickNewOptionEpsilonGreedily(state, epsilon)
        current_option=self.options[self.current_option_idx]
        action, action_number = current_option.pickAction(state)
        # Record action number corresponding to last action taken
        self.last_action_taken = action_number
        
        return action,action_number
    
    def recordTransition(self, state, action_number,reward, next_state,learning_flag=True,switch_reward=True):
        # Add reward discounted by current discounting factor
        if learning_flag:
            self.rewards.append(reward) #+= (self.gamma ** self.k) * reward
            self.trajectory.append(next_state) 
            self.pis.append(action_number)          

        current_option=self.options[self.current_option_idx]

        #if learning_flag:
            #self._updateQValuesIntra_core(state, reward, action_number,next_state)
        if current_option.beta[next_state] == 1:
            if learning_flag:
                # propogate back whole trajectory
                self._updateQValueOption()
                if switch_reward:
                    # intra option per action update
                    self._updateQValuesIntra()

            self._resetCurrentOption()
    
    def run_episode(self,env,max_stepnum=1000,interruption=True,verbose=False,start_new=True,learning_flag=True,switch_reward=True):

        if learning_flag:
            epsilon=1
        else:
            epsilon=0.05

        if start_new:
            start_idx,x,y=env.chooseRandomState()
            env.defineStartState(start_idx)
        env.reset()
        

        n_steps = 0
        cumulative_reward = 0
        env.reset()
        goal_idx=env.getGoalState()
        
        done=False
        while True:

            n_steps += 1
            state=env._getStateIndex(env.currX,env.currY)

            #if not learning_flag:
            #    action,action_number = self.epsilonGreedyPolicy_Interruption(state,epsilon=epsilon)
            #else:
            action,action_number = self.epsilonGreedyPolicy(state,epsilon=epsilon)
            
            if verbose:
                print("State = {}, Option = {}, Action = {}".format(
                    state, self.current_option_idx, action))
                
            reward=env.act_option(action)
            cumulative_reward+=reward
            
            next_state=env._getStateIndex(env.currX,env.currY)
            if next_state==goal_idx:
                done=True
            if done and not learning_flag:
                self._resetCurrentOption()
                return n_steps,1
 
            self.recordTransition(state, action_number,reward, next_state,learning_flag,switch_reward)
            
            state = next_state
            if done:
                if start_new:    
                    start_idx,x,y=env.chooseRandomState()
                    env.defineStartState(start_idx)
                self._resetCurrentOption()
                env.reset()
            if n_steps>max_stepnum:
                break
        return n_steps,cumulative_reward