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
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.values = util.Counter()

    def getQValue(self, state, action):
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        actions = self.getLegalActions(state)
        return max((self.getQValue(state, action) for action in actions),
                   default=0.0)

    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None

        best_value = self.computeValueFromQValues(state)
        best_actions = []
        for action in actions:
            if self.getQValue(state, action) == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def getAction(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        return self.computeActionFromQValues(state)

        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        old_value = self.getQValue(state, action)
        next_value = self.computeValueFromQValues(nextState)
        target = reward + self.discount * next_value

        self.values[(state, action)] = old_value + self.alpha * (target - old_value)

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
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        qValue = 0.0

        for feature, value in features.items():
            qValue += self.weights[feature] * value

        return qValue

    def update(self, state, action, nextState, reward):
        target = reward + self.discount * self.computeValueFromQValues(nextState)
        difference = target - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)

        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
