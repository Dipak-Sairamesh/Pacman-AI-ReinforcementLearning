# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        """
        Commands to run:
        python gridworld.py -a q -k 5 -m
        python autograder.py -q q6
        """
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        if (state, action) in self.qvalues.keys(): # For states seen before
            return self.qvalues[(state, action)] # Q node values
        else:
            return 0.0 # For states never seen before

        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        actions = self.getLegalActions(state)
        qvalues = [self.getQValue(state, action) for action in actions] # List of Q values
        if not len(qvalues):
            return 0.0
        return max(qvalues) # Return max action

        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)
        actions = [] # List for choosing the best action

        if len(legalActions) == 0:
            return None

        value = self.computeValueFromQValues(state)
        for action in legalActions:
            if value == self.getQValue(state, action):
                actions.append(action)
        # Break ties by picking randomly
        return random.choice(actions)

        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)

          Commands to run:
          python gridworld.py -a q -k 100
          python gridworld.py -a q -k 100 --noise 0.0 -e 0.1
          python gridworld.py -a q -k 100 --noise 0.0 -e 0.9
          python autograder.py -q q7
          python crawler.py

        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        self.qvalues[(state, action)] = (1-self.alpha) * (self.getQValue(state, action)) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes

        Commands to run:
        python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
        python autograder.py -q q9
        python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.

       Commands to run:
       python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid
       python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
       python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
       python autograder.py -q q10

    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)
        return weights * features

        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        features = self.featExtractor.getFeatures(state, action)
        difference = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        for feature in features: # Each weight is associated with a particular feature
            self.weights[feature] += self.alpha * difference * features[feature]

        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
