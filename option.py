import numpy as np

class Option():
    def __init__(self, option):
        self.name = option
        # Set I (initiation set), beta (termination set), pi (policy) 
        # all are of length env.numStates
    def setIBetaPi(self,I,beta,pi):
        self.I=I
        self.beta=beta
        self.pi=pi

    def pickAction(self, state):
        action_number = self.pi[state]
        if action_number == 0:
            action = "up"
        elif action_number == 1:
            action = "right"
        elif action_number == 2:
            action = "down"
        elif action_number == 3:
            action = "left"
        elif action_number == 4:
            action = "terminate"
        # Return action number, used for intra-option model learning
        return action, action_number
               
    def visualize(self):
        plt.imshow(self.I)
        plt.colorbar()
        plt.title("Initiation set")
        plt.show()
        plt.imshow(self.beta)
        plt.colorbar()
        plt.title("Termination set")
        plt.show()
        plt.imshow(self.pi)
        plt.colorbar()
        plt.title("Policy")
        plt.show()
        

        
    def __str__(self):
        return self.name