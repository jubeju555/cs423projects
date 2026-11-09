# valueIterationAgents.py
# ----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

from __future__ import absolute_import
import mdp
import util

from learningAgents import ValueEstimationAgent
from six.moves import range


class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent that plans from a known MDP before acting.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()

        # Use a separate counter for each pass so every update uses the
        # values from the previous iteration.
        for _ in range(iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)
                if actions:
                    newValues[state] = max(
                        self.computeQValueFromValues(state, action)
                        for action in actions
                    )
            self.values = newValues

    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        total = 0.0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(
                state, action):
            reward = self.mdp.getReward(state, action, nextState)
            total += probability * (reward + self.discount * self.values[nextState])
        return total

    def computeActionFromValues(self, state):
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None

        bestAction = None
        bestValue = float("-inf")
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
