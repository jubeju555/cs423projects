# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    Choose a move based on food and nearby ghosts.
    """

    def getAction(self, gameState: GameState):
        legalMoves = gameState.getLegalActions()

        scores = [
            self.evaluationFunction(gameState, action)
            for action in legalMoves
        ]

        bestScore = max(scores)
        bestIndices = [
            i for i in range(len(scores))
            if scores[i] == bestScore
        ]

        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Favor food, avoid active ghosts, and approach scared ghosts
        when their timers make capture plausible.
        """
        afterMove = currentGameState.generatePacmanSuccessor(action)

        if afterMove.isWin():
            return float("inf")
        if afterMove.isLose():
            return float("-inf")

        pacSpot = afterMove.getPacmanPosition()
        foodLeft = afterMove.getFood().asList()
        ghostList = afterMove.getGhostStates()
        moveScore = afterMove.getScore()

        # Less food left is better.
        moveScore -= len(foodLeft) * 4

        foodGap = float("inf")
        for pellet in foodLeft:
            gap = manhattanDistance(pacSpot, pellet)
            if gap < foodGap:
                foodGap = gap

        if foodLeft:
            moveScore += 10 / (foodGap + 1)

        for ghost in ghostList:
            ghostGap = manhattanDistance(pacSpot, ghost.getPosition())

            if ghost.scaredTimer > 0:
                if ghostGap < ghost.scaredTimer:
                    moveScore += 20 / (ghostGap + 1)
            else:
                if ghostGap == 0:
                    return float("-inf")

                # Avoid ending a move beside an active ghost.
                if ghostGap <= 1:
                    moveScore -= 200

                moveScore -= 12 / ghostGap

        if action == Directions.STOP:
            moveScore -= 2

        return moveScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Pacman picks the highest score, and ghosts pick the lowest.
    """

    def getAction(self, gameState: GameState):
        def searchTurns(board, whoseTurn, roundsDone):
            # Evaluate the board when the game or search is finished.
            if board.isWin() or board.isLose():
                return self.evaluationFunction(board), Directions.STOP

            if roundsDone >= self.depth:
                return self.evaluationFunction(board), Directions.STOP

            movesHere = board.getLegalActions(whoseTurn)

            if not movesHere:
                return self.evaluationFunction(board), Directions.STOP

            turnAfter = whoseTurn + 1
            roundsAfter = roundsDone

            # Count a full round after everyone has moved.
            if turnAfter == board.getNumAgents():
                turnAfter = 0
                roundsAfter += 1

            moveScores = []

            for move in movesHere:
                nextBoard = board.generateSuccessor(whoseTurn, move)

                result = searchTurns(
                    nextBoard,
                    turnAfter,
                    roundsAfter
                )
                moveScores.append(result[0])

            if whoseTurn == 0:
                pickedScore = max(moveScores)
            else:
                pickedScore = min(moveScores)

            # index picks the first move if several scores are tied.
            pickedIndex = moveScores.index(pickedScore)
            pickedMove = movesHere[pickedIndex]

            return pickedScore, pickedMove

        answer = searchTurns(gameState, 0, 0)
        return answer[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax with alpha-beta pruning across all ghost turns.
    """

    def getAction(self, gameState: GameState):
        def searchTurns(board, whoseTurn, roundsDone, alpha, beta):
            if board.isWin() or board.isLose():
                return self.evaluationFunction(board), Directions.STOP

            if roundsDone >= self.depth:
                return self.evaluationFunction(board), Directions.STOP

            movesHere = board.getLegalActions(whoseTurn)

            if not movesHere:
                return self.evaluationFunction(board), Directions.STOP

            turnAfter = whoseTurn + 1
            roundsAfter = roundsDone

            # A round finishes after the last ghost moves.
            if turnAfter == board.getNumAgents():
                turnAfter = 0
                roundsAfter += 1

            pickedMove = movesHere[0]

            if whoseTurn == 0:
                pickedScore = float("-inf")

                for move in movesHere:
                    nextBoard = board.generateSuccessor(whoseTurn, move)
                    result = searchTurns(
                        nextBoard, turnAfter, roundsAfter, alpha, beta
                    )
                    moveScore = result[0]

                    if moveScore > pickedScore:
                        pickedScore = moveScore
                        pickedMove = move

                    # Equality must stay in the search.
                    if pickedScore > beta:
                        return pickedScore, pickedMove

                    alpha = max(alpha, pickedScore)

            else:
                pickedScore = float("inf")

                for move in movesHere:
                    nextBoard = board.generateSuccessor(whoseTurn, move)
                    result = searchTurns(
                        nextBoard, turnAfter, roundsAfter, alpha, beta
                    )
                    moveScore = result[0]

                    if moveScore < pickedScore:
                        pickedScore = moveScore
                        pickedMove = move

                    if pickedScore < alpha:
                        return pickedScore, pickedMove

                    beta = min(beta, pickedScore)

            return pickedScore, pickedMove

        answer = searchTurns(
            gameState, 0, 0, float("-inf"), float("inf")
        )
        return answer[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, agentIndex, depth):
            # Stop after the requested number of complete Pacman turns, or
            # immediately when the game has already ended.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()

            # A Pacman node chooses the action with the highest expected
            # value.  Ties are intentionally resolved by keeping the first
            # legal action, which matches the project framework's convention.
            if agentIndex == self.index:
                bestValue = float("-inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    nextDepth = depth + 1 if nextAgent == self.index else depth
                    bestValue = max(
                        bestValue,
                        expectimax(successor, nextAgent, nextDepth)
                    )
                return bestValue

            # Ghosts are chance nodes: every legal action is equally likely,
            # so their value is the arithmetic mean of successor values.
            totalValue = 0.0
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                nextDepth = depth + 1 if nextAgent == self.index else depth
                totalValue += expectimax(successor, nextAgent, nextDepth)
            return totalValue / len(legalActions)

        bestAction = None
        bestValue = float("-inf")
        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, action)
            value = expectimax(successor, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
