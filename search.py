# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.
    """
    # Using a stack since dfs = LIFO
    myStack = util.Stack()
    startState = problem.getStartState()
    # Pushing a tuple of (state, list of actions to get here so far)
    myStack.push((startState, []))

    visited = []

    while not myStack.isEmpty():
        current, actionsSoFar = myStack.pop()

        # Check if we're done FIRST before doing anything else
        if problem.isGoalState(current):
            return actionsSoFar

        # Only expand if we haven't seen it before
        if current not in visited:
            visited.append(current)

            successors = problem.getSuccessors(current)
            for succ in successors:
                nextState = succ[0]
                actionToTake = succ[1]
                # We do not care about cost for dfs so just ignoring 3rd value

                if nextState not in visited:
                    # Make a NEW list because using the same one messes
                    # up the other branches since lists are mutable
                    updatedActions = actionsSoFar + [actionToTake]
                    myStack.push((nextState, updatedActions))

    # If the loop ends and we never hit return, no path exists
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    fringe = util.Queue()
    start = problem.getStartState()
    fringe.push((start, []))
    visited = {start}

    while not fringe.isEmpty():
        state, actions = fringe.pop()
        if problem.isGoalState(state):
            return actions

        for successor, action, _ in problem.getSuccessors(state):
            if successor not in visited:
                visited.add(successor)
                fringe.push((successor, actions + [action]))

    return []

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # Using a priority queue so we always expand the cheapest path first
    pq = util.PriorityQueue()
    startState = problem.getStartState()
    # Pushing (state, path, cost so far) - priority is just the cost
    pq.push((startState, [], 0), 0)

    # Keep track of states we already expanded
    visited = []

    while not pq.isEmpty():
        currState, path, currCost = pq.pop()

        # Goal check happens when we pop, not when we push
        if problem.isGoalState(currState):
            return path

        # Only expand if we haven't already dealt with this state
        if currState not in visited:
            visited.append(currState)

            for nextState, action, stepCost in problem.getSuccessors(currState):
                if nextState not in visited:
                    # Build up the new path and total cost
                    newPath = path + [action]
                    newCost = currCost + stepCost
                    pq.push((nextState, newPath, newCost), newCost)

    # If we get here the queue emptied out without finding a goal
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
