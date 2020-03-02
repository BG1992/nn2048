import nn2048helper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(profile='full')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.convH1 = nn.Conv2d(in_channels=18, out_channels=18*4, kernel_size=(4,1), stride=1, padding=(3,0))
        self.convV1 = nn.Conv2d(in_channels=18, out_channels=18*4, kernel_size=(1,4), stride=1, padding=(0,3))
        self.convS1 = nn.Conv2d(in_channels=18, out_channels=18*4, kernel_size=(2,2), stride=1, padding=(1,1))

        self.convH2 = nn.Conv2d(in_channels=18*4, out_channels=18*4*4, kernel_size=(2,2), stride=1, padding=(1,1))
        self.convV2 = nn.Conv2d(in_channels=18*4, out_channels=18*4*4, kernel_size=(2,2), stride=1, padding=(1,1))
        self.convS2 = nn.Conv2d(in_channels=18*4, out_channels=18*4*4, kernel_size=(2,2), stride=1, padding=(1,1))

        self.fcH1 = nn.Linear(in_features=18*4*4*40, out_features=18*4)
        self.fcV1 = nn.Linear(in_features=18*4*4*40, out_features=18*4)
        self.fcS1 = nn.Linear(in_features=18*4*4*36, out_features=18*4)

        self.fcMerged = nn.Linear(18*4*3, out_features=18*4*3)
        self.out = nn.Linear(in_features=18*4*3, out_features=1)

    def forward(self, x):
        xH = self.convH1(x)
        xH = F.relu(xH)
        xH = self.convH2(xH)
        xH = F.relu(xH)
        xH = xH.view(-1, 18*4*4*40)
        xH = self.fcH1(xH)
        xH = F.relu(xH)

        xV = self.convV1(x)
        xV = F.relu(xV)
        xV = self.convV2(xV)
        xV = F.relu(xV)
        xV = xV.view(-1, 18*4*4*40)
        xV = self.fcV1(xV)
        xV = F.relu(xV)

        xS = self.convS1(x)
        xS = F.relu(xS)
        xS = self.convV2(xS)
        xS = F.relu(xS)
        xS = xS.view(-1, 18*4*4*36)
        xS = self.fcS1(xS)
        xS = F.relu(xS)

        xMerged = torch.cat([xH, xV, xS], dim = 1)
        xMerged = self.fcMerged(xMerged)
        xMerged = F.relu(xMerged)

        return self.out(xMerged)

def getScore(board, Net=None, boardTensor = None):
    if len(nn2048helper.checkMoves(board)) == 0:
        return 0
    if Net is not None:
        if boardTensor is not None:
            score = Net.forward(boardTensor.view(-1, 18, 4, 4)).data.tolist()[0][0]
            return score
        score = Net.forward(nn2048helper.makeTensor(board).view(-1, 18, 4, 4)).data.tolist()[0][0]
        return score
    return 0

def expectiDepthMax(board, Net, gamma, currDepth, finalDepth):
    if currDepth >= finalDepth:
        maxScoreSoFar = 0
        movesToDo = nn2048helper.checkMoves(board)
        chosenMove = None
        for candMove in movesToDo:
            candBoard, candRewToNet, candReward = nn2048helper.makeMove(board, candMove)
            candBoardTensor = nn2048helper.makeTensor(candBoard)
            candScore = getScore(candBoard, Net, candBoardTensor)
            if candRewToNet + gamma*candScore > maxScoreSoFar:
                maxScoreSoFar = candRewToNet + gamma*candScore
                chosenMove = candMove
        return maxScoreSoFar, chosenMove
    else:
        maxScoreSoFar = 0
        movesToDo = nn2048helper.checkMoves(board)
        chosenMove = None
        for candMove in movesToDo:
            candBoard, candRewToNet, candReward = nn2048helper.makeMove(board, candMove)
            candScore = 0
            candNextBoards = nn2048helper.addAllPossibleNums(candBoard)
            for candNextBoard in candNextBoards:
                nonZeroCt = np.count_nonzero(candNextBoard[0])
                candScore += gamma*expectiDepthMax(candNextBoard[0], Net, gamma, currDepth+1,
                                                   depthsPolicy[16-nonZeroCt])[0]*candNextBoard[1]
            candScore *= 2/len(candNextBoards)
            if candRewToNet + gamma*candScore > maxScoreSoFar:
                maxScoreSoFar = candRewToNet + gamma*candScore
                chosenMove = candMove
        return maxScoreSoFar, chosenMove

class Game():
    def __init__(self, Net, gamma):
        self.Net = Net
        self.gamma = gamma
    def run(self, fileScoresPath, gameNum, depthsPolicy):
        overallScore = 0
        board = nn2048helper.initBoard()
        currMax = 0
        print(board)
        while True:
            currMax = max(currMax, np.max(board))
            scoreAfterMove, moveToDo = expectiDepthMax(board, self.Net, self.gamma, 1, depthsPolicy[16-np.count_nonzero(board)])
            if moveToDo is not None:
                board, rewToNet, reward = nn2048helper.makeMove(board, moveToDo)
                overallScore += reward
                print(board, scoreAfterMove, overallScore)
                board = nn2048helper.addNum(board)
            else: break
        print('Score: ', overallScore)
        f = open(fileScoresPath, 'a')
        f.write(str(gameNum) + ', ' + str(overallScore) + ', ' + str(currMax) + '\n')
        f.close()

depthsPolicy = [3,3,2,2] + [1]*12
#depthsPolicy = [4,3,3,3,2,2,2,1,1,1,1,1,1,1,1]
fileScoresPath = 'D://Network2048//mainV3//fileScoresMainRun.txt'
model = Net()
model.load_state_dict(torch.load('D://Network2048//mainV3//model2048NetworkFinal.pth'))
gamma = 0.97

currGame = Game(model, gamma)
for gameNum in range(1, 101):
    currGame.run(fileScoresPath, gameNum, depthsPolicy)