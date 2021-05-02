import sys
import math
import warnings
import numpy as np
import matplotlib.pylab as plt

from Learning import Learning
from Drawing import Plotter
from Utils import Utils
from Utils import ArgsParser
from Environment import GridWorld
from MDPStats import MDPStats
from main import discoverOptions
from QLearning import QLearning
from Drawing import Plotter
from option import Option

def discover_PRSR(env_path,pseudo_reward=None,gamma_learn=0.9,gamma_PR=0.9,simulate_num=200):
    env = GridWorld(path = env_path, useNegativeRewards=False)
    numRows, numCols = env.getGridDimensions()
    start_idx=env._getStateIndex(env.startX, env.startY)
    PR_ave=np.zeros((numRows,numCols))
    SR_ave=np.zeros((numRows,numCols))

    goals=[]
    PR_s=[]
    SR_s=[]
    for simulate_idx in range(simulate_num):

        # creat a new environment
        env = GridWorld(path = env_path, useNegativeRewards=False)
        numStates = env.getNumStates()

        if pseudo_reward is None:
            # randomly select a goal (need while loop as the random goal could be a wall)
            while True:
                goal_idx = np.random.choice(np.arange(numStates))
                if env.defineGoalState(int(goal_idx)):
                    break
            goals.append(int(goal_idx))
            # prepare to learn policy iteration
            polIter = Learning(gamma_learn, env, augmentActionSet=True)
            V, pi = polIter.solvePolicyIteration()   

            # get forward one-step transition probability
            P_pro,indeces = get_Ppro(env,pi,[goal_idx])
            SR,PR,SR_contingency,PR_contingency,_ = get_SRPR(P_pro,gamma=gamma_PR)
            PR_ = visualize_SRPR(env,PR,goal_idx)
            SR_ = visualize_SRPR(env,SR,goal_idx)  
        else: 
            # use non-sparse pseudo-reward
            env.defineRewardFunction(pseudo_reward[:,simulate_idx])
            polIter = Learning(gamma_learn, env, augmentActionSet=True)
            V, pi = polIter.solvePolicyIteration()

            # get forward one-step transition probability
            P_pro,indeces = get_Ppro(env,pi,[])
            SR,PR,SR_contingency,PR_contingency,_ = get_SRPR(P_pro,gamma=gamma_PR)

            PR_ = visualize_vec(env,np.matmul(PR,pseudo_reward[indeces,simulate_idx]))
            SR_ = visualize_vec(env,np.matmul(SR,pseudo_reward[indeces,simulate_idx]))

        PR_s.append(PR_)
        SR_s.append(SR_)
        PR_ave = PR_ave + PR_/simulate_num
        SR_ave = SR_ave + SR_/simulate_num
    '''
    # now detect peaks
    detected_peaks=detect_peaks(PR_ave)
    local_max=np.argwhere(detected_peaks)
    # subscript to indices
    ind_all=[]
    for i in range(np.shape(local_max)[0]):
        ind=local_max[i,0]*13+local_max[i,1]
        ind_all.append(ind)
    ind_all=np.array(ind_all)
    '''
    
    return PR_ave,SR_ave,indeces

def findOptionStartEnd(env_path,PR,optionNum,starts=None):
    # 0) Start from a random state s, that has index start_idx
    # learn random walk's SR
    env = GridWorld(path = env_path, useNegativeRewards=False)
    indeces = findNoWallState(env)
    numStates=env.numStates

    starts_subgoals_x_y=[]

    P=get_Ppro_from_adjacency(env)
    SR_random,PR_random,SR_contingency,PR_contingency,_ = get_SRPR(P,gamma=0.99)

    if starts is None:
        start_idx,x,y=env.chooseRandomState()

    for o_ind in range(2*optionNum):
        #print('x,y',x,y)
        #print(o_ind)
        if starts is not None:
            x,y=starts[o_ind]
            start_idx=env._getStateIndex(x,y)

        SR_random_start = visualize_vec(env,SR_random[np.argwhere(indeces==start_idx)[0][0],:])
        SR_PR=SR_random_start*PR
        SR_PR[x,y]=0 #get rid of itself

        subgoal_ideces=np.argsort(SR_PR.reshape(-1,1).ravel())[::-1]
        if starts is None:
            # auto detect mode
            while True:
                subgoal_idx=subgoal_ideces[0] #max
                subgoalX,subgoalY = env.getStateXY(subgoal_idx)

                previous_explored=(start_idx,subgoal_idx,x,y,subgoalX,subgoalY) in starts_subgoals_x_y
                #print('(start_idx,subgoal_idx,x,y,subgoalX,subgoalY)',(start_idx,subgoal_idx,x,y,subgoalX,subgoalY))
                #print('previous_explored',previous_explored)
                if not previous_explored:
                    starts_subgoals_x_y.append((start_idx,subgoal_idx,x,y,subgoalX,subgoalY))
                    start_idx=subgoal_idx
                    x,y=subgoalX,subgoalY
                    break
                else:
                    start_idx,x,y=env.chooseRandomState()
                    break
        else:
            # simple heuristic find the first two highest
            for t in range(2):
                subgoal_idx=subgoal_ideces[t]
                subgoalX,subgoalY = env.getStateXY(subgoal_idx)
                starts_subgoals_x_y.append((start_idx,subgoal_idx,x,y,subgoalX,subgoalY))

        if len(starts_subgoals_x_y)>=optionNum:
            break
    return starts_subgoals_x_y,SR_random

def findEigenReward(env_path,outputPath=None):
    discoverNegation=0

    env = GridWorld(path = env_path, useNegativeRewards=False)
    numStates = env.getNumStates()
    W = env.getAdjacencyMatrix()
    D = np.zeros((numStates, numStates))

    # Obtaining the Valency Matrix
    for i in range(numStates):
        for j in range(numStates):
            D[i][i] = np.sum(W[i])

        # Making sure our final matrix will be full rank
        for i in range(numStates):
            if D[i][i] == 0.0:
                D[i][i] = 1.0

    # Normalized Laplacian
    L = D - W
    expD = Utils.exponentiate(D, -0.5)
    normalizedL = expD.dot(L).dot(expD)
        
    # Eigendecomposition
    # IMPORTANT: The eigenvectors are in columns
    eigenvalues, eigenvectors = np.linalg.eig(normalizedL)
    # I need to sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # If I decide to use both directions of the eigenvector, I do it here.
    # It is easier to just change the list eigenvector, even though it may
    # not be the most efficient solution. The rest of the code remains the same.
    if discoverNegation:
        oldEigenvalues = eigenvalues
        oldEigenvectors = eigenvectors.T
        eigenvalues = []
        eigenvectors = []
        for i in range(len(oldEigenvectors)):
            eigenvalues.append(oldEigenvalues[i])
            eigenvalues.append(oldEigenvalues[i])
            eigenvectors.append(oldEigenvectors[i])
            eigenvectors.append(-1 * oldEigenvectors[i])

            eigenvalues = np.asarray(eigenvalues)
            eigenvectors = np.asarray(eigenvectors).T

    # Plotting all the basis
    if outputPath is not None:
        plot = Plotter(outputPath, env)
        plot.plotBasisFunctions(eigenvalues, eigenvectors)
    for e in range(len(eigenvalues)):
        eigenvectors[:,e]=eigenvectors[:,e]/np.linalg.norm(eigenvectors[:,e])#*eigenvalues[e]
    return eigenvectors

def discover_option_policy(env_path,figpath,starts_subgoals_x_y,SR_random,PR,SR_thresh=0.1):
    gamma_learn=0.9
    options=[]
    env = GridWorld(path = env_path, useNegativeRewards=False)
    indeces = findNoWallState(env)
    
    for o_ind in range(len(starts_subgoals_x_y)):
    # 1) Look for a policy to go to a state s' with highest +(SR(s|.)*E(PR(.|reward))
    #    (1) [beta] Extract s' (a subgoal)
        start_idx,subgoal_idx,x,y,subgoalX,subgoalY=starts_subgoals_x_y[o_ind]

        SR_random_start = visualize_vec(env,SR_random[np.argwhere(indeces==start_idx)[0][0],:])
        SR_PR=SR_random_start*PR
    
        #   (2) [I] Learn from a subset of states to get to subgoal: these states form I
        I_ind = np.argwhere(SR_random_start.reshape((-1,1))>=SR_thresh)[:,0]
        I = np.zeros(env.numStates)
        I[I_ind]=1
    
        #   (3) [pi] find policy
        #       creat a new environment
        env1 = GridWorld(path = env_path, useNegativeRewards=False)
        numStates = env1.getNumStates()
        env1.defineGoalState(subgoal_idx)
        env1.defineStartState(start_idx) # must call defineStartState after defineGoalState

        #       learn by policy iteration
        polIter = Learning(gamma_learn, env1, augmentActionSet=True)
        V, pi = polIter.solvePolicyIteration_subsetStates(np.argwhere(I).ravel(),theta=0.001)
        V=V[:numStates]
        pi=pi[:numStates]
        beta=np.zeros(env.numStates)+1
        beta[I_ind]=0
        beta[pi==4]=1

        #  (4) put it into option
        current_option=Option(str(subgoal_idx))
        current_option.setIBetaPi(I,beta,pi)
        options.append(current_option)
    
        #  (5) plotting
        if len(figpath)>0:
            plt.subplot(1,4,1)
            plt.imshow(SR_random_start)
            plt.plot(y,x,lw=0, marker='s',color='red',fillstyle='none')
            plt.title('SR(.|('+str(x)+','+str(y)+'))')
            plt.subplot(1,4,2)
            plt.imshow(SR_PR)
            plt.title('SR(.|('+str(x)+','+str(y)+'))*E(PR(.|r))'+'\n next subgoal:('+str(subgoalX)+','+str(subgoalY)+')')
            plt.subplot(1,4,3)
            Pn=np.reshape(V,(env1.numRows,env1.numCols),order='C')
            plt.imshow(env1.matrixMDP+Pn)
            plt.subplot(1,4,4)
            Pn=np.reshape(pi,(env1.numRows,env1.numCols),order='C')
            plt.imshow(env1.matrixMDP+Pn)   
            plt.savefig(figpath+str(o_ind)+'.png')
            plt.cla()
    return options

def learnOptionEigen(env_path,pseudoreward):
    env = GridWorld(path = env_path, useNegativeRewards=False)
    numStates=env.numStates
    env.defineRewardFunction(pseudoreward)
    indeces=findNoWallState(env)
    
    subgoal_idx=np.argmax(pseudoreward)
    I=np.zeros(numStates)
    I[indeces]=1
    I[subgoal_idx]=0

    polIter = Learning(0.9, env, augmentActionSet=True)
    V, pi = polIter.solvePolicyIteration_subsetStates(np.argwhere(I).ravel(),theta=0.001)
    pi=pi[:numStates]
    
    beta=np.zeros(numStates)+1
    beta[indeces]=0
    beta[pi==4]=1
    
    current_option=Option(str(subgoal_idx))
    current_option.setIBetaPi(I,beta,pi)
    
    return current_option

def add_premitive_actions(env_path):
    premi_options=[]
    
    env = GridWorld(path = env_path, useNegativeRewards=False)
    
    # add premitive actions
    premitive_actions=env.getActionSet()
    env.defineGoalState(None)
    action_id=0
    for a in premitive_actions:
        option=Option(str(a))
        I=np.ones(env.numStates)
        beta=np.ones(env.numStates)
        pi=np.ones(env.numStates)*action_id

        for s in range(env.numStates):
            success1=env.defineStartState(s)
            if not success1:
                I[s]=False
                continue

            reward=env.act(a)
            fail2=s==env._getStateIndex(env.currX,env.currY)

            if fail2:
                I[s]=False
        option.setIBetaPi(I,beta,pi)
        action_id+=1
        premi_options.append(option)
    return premi_options

def get_Ppro(env,pi,goal_idx):
    numStates = env.getNumStates()
    actionset=env.getActionSet()
    P=np.zeros((numStates,numStates))
    
    # fill in transitions
    for idx in range(numStates):
        i, j = env.getStateXY(idx)
        if env.matrixMDP[i,j]!=-1:
            if pi[idx]<4:
                action=actionset[pi[idx]]
                nextStateIdx, reward=env.getNextStateAndReward(idx, action)
            else:
                nextStateIdx=idx
            P[idx,nextStateIdx] = P[idx,nextStateIdx]+1
    
    # if goal_idx is not provided, figure out by finding pure absorbing state
    if len(goal_idx)==0:
        goal_idx=np.argwhere(np.diag(P)==1).ravel()
            
    # fill in random actions for goal state so that the chain is non-absorbing
    indeces=[] # non-wall states
    for idx in range(numStates):
        i, j = env.getStateXY(idx)
        if env.matrixMDP[i,j]!=-1:
            P[goal_idx,idx]=1
            indeces.append(idx)

    if len(goal_idx)>0:
        for g in goal_idx:
            P[g,:]=P[g,:]/np.sum(P[g,:])
    P_pro=P[indeces,:]
    P_pro=P_pro[:,indeces]
    
    return P_pro,indeces

def get_Ppro_from_adjacency(env):
    numStates = env.getNumStates()
    
    # fill in transitions
    P = env.getAdjacencyMatrix()
            
    # fill in random actions for goal state so that the chain is non-absorbing
    indeces=findNoWallState(env)
    
    P_pro=P[indeces,:]
    P_pro=P_pro[:,indeces]
    
    P_normalized=P_pro/np.sum(P_pro,1)[:,None]
    
    return P_normalized

def get_SRPR(P_pro,gamma=0.9):
    
    
    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    eig_value, eig_vectors = np.linalg.eig(P_pro.T)
    evec1 = eig_vectors[:,np.isclose(eig_value, 1)]
    evec1 = evec1.ravel()

    stationary = evec1 / evec1.sum()

    #get rid of the 0j imagainary
    P_stationary = stationary.real
    
    D=np.diag(P_stationary)
    P_retro = np.matmul(np.matmul(D,P_pro),np.linalg.inv(D)); #D*P_pro*D^-1

    I=np.eye(np.shape(P_pro)[0]);
    SR=np.matmul(P_pro,np.linalg.inv(I-gamma*P_pro));
    SR_marginal = np.mean(SR,0);
    SR_contingency = SR - SR_marginal;
    PR=np.matmul(P_retro,np.linalg.inv(I-gamma*P_retro));
    PR_marginal = np.mean(PR,1);
    PR_contingency = PR - PR_marginal;
    
    return SR,PR,SR_contingency,PR_contingency,P_stationary

def visualize_SRPR(env,PR,goal_idx):
    # return a matrix PR(.|reward) so that you could plot as plt.imshow(PR)
    numRows, numCols = env.getGridDimensions()
    numStates = env.getNumStates()
    
    indeces=findNoWallState(env)
    
    indeces=np.array(indeces)
    PR_=np.zeros((numRows,numCols))
    ind_reward=np.argwhere(indeces==goal_idx)[0][0]
    for idx in range(len(indeces)):
        i, j = env.getStateXY(indeces[idx])
        PR_[int(i),int(j)]=PR[idx,ind_reward]
    return PR_

def visualize_SRPR_vector(env,PR):
    # return a matrix PR so that you could plot as plt.imshow(PR)
    numRows, numCols = env.getGridDimensions()
    numStates = env.getNumStates()
    
    indeces=findNoWallState(env)
    
    indeces=np.array(indeces)
    PR_=np.zeros((numRows,numCols))
    for idx in range(len(indeces)):
        i, j = env.getStateXY(indeces[idx])
        PR_[int(i),int(j)]=PR[idx]
    return PR_

def visualize_vec(env,vec):
    # return a matrix PR(.|reward) so that you could plot as plt.imshow(PR)
    numRows, numCols = env.getGridDimensions()
    numStates = env.getNumStates()
    
    indeces=findNoWallState(env)
    
    indeces=np.array(indeces)
    PR_=np.zeros((numRows,numCols))
    for idx in range(len(indeces)):
        i, j = env.getStateXY(indeces[idx])
        PR_[int(i),int(j)]=vec[idx]
    return PR_

def findNoWallState(env):
    numStates=env.numStates
    indeces=[] # non-wall states
    for idx in range(numStates):
        i, j = env.getStateXY(idx)
        if env.matrixMDP[i,j]!=-1:
            indeces.append(idx)
    return np.array(indeces)

