import nn2048helper
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import defaultdict
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

def makeMoveUnderPolicy(board, Net, gamma, eps):
    movesToDo = nn2048helper.checkMoves(board)
    if len(movesToDo) == 0:
        return None, 0, 0
    if random.random() <= eps:
        move = random.choice(movesToDo)
        newBoard, rewToNet, reward = nn2048helper.makeMove(board, move)
        return newBoard, rewToNet, reward
    else:
        maxScoreSoFar = float('-inf')
        for candMove in movesToDo:
            candBoard, candRewToNet, candReward = nn2048helper.makeMove(board, candMove)
            candBoardTensor = nn2048helper.makeTensor(candBoard)
            candScore = getScore(candBoard, Net, candBoardTensor)
            if candRewToNet + gamma*candScore > maxScoreSoFar:
                maxScoreSoFar = candRewToNet + gamma*candScore
                chosenBoard = candBoard.copy()
                chosenRewToNet = candRewToNet
                chosenReward = candReward
        return chosenBoard, chosenRewToNet, chosenReward

def newGain(currBoardAfterMove, Net, gamma):
    #expected value taken from function new gains: r + gamma*V(s')
    tempNewGain = 0
    allPossibleBoards = nn2048helper.addAllPossibleNums(currBoardAfterMove)
    for newBoard, newProb in allPossibleBoards:
        newBoardAfterMove, newRewToNet, newReward = makeMoveUnderPolicy(newBoard, Net, gamma, 0)
        tempNewGain += ((newRewToNet + gamma*getScore(newBoardAfterMove, Net))*newProb)
    tempNewGain *= (2/len(allPossibleBoards))
    return tempNewGain

class Batch():
    def __init__(self, eps, batchSize):
        self.batchData = defaultdict(list)
        self.eps = eps
        self.batchSize = batchSize
        self.memory = 5000
    def run(self, Net, alpha, gamma, epNum):
        while len(self.batchData) <= self.memory:
            newGame = Game(self.eps, Net, alpha, gamma, epNum)
            newGame.run()
            for pos in newGame.data:
                self.batchData[pos[0]].append(pos[1])
            print(len(self.batchData))
        batchDataToTrain = []
        for pos in self.batchData:
            batchDataToTrain.append([pos, sum(self.batchData[pos])/len(self.batchData[pos])])
        random.shuffle(batchDataToTrain)
        xTrain, yTrain = [], []
        for i in range(0, len(self.batchData) - self.batchSize + 1, self.batchSize):
            xTrain.append(torch.stack(list(map(lambda x: x[0], batchDataToTrain[i:i + self.batchSize]))))
            yTrain.append(torch.stack(list(map(lambda x: torch.tensor(x[1]), batchDataToTrain[i:i + self.batchSize]))))
        return xTrain, yTrain

class Game():
    def __init__(self, eps, Net, alpha, gamma, epNum):
        self.eps = eps
        self.data = []
        self.Net = Net
        self.alpha = alpha
        self.gamma = gamma
        self.epNum = epNum
    def run(self):
        overallScore = 0
        if self.epNum <= 50:
            board = nn2048helper.initBoard()
        else:
            if random.random() <= 0.3:
                board = nn2048helper.initBoard()
            else:
                board = nn2048helper.randomBoard(random.randint(2, 12))
        board, rewToNet, reward = makeMoveUnderPolicy(board, self.Net, self.gamma, self.eps)
        overallScore += reward
        currMax = 0
        while board is not None:
            currMax = max(currMax, np.max(board))
            boardNewGain = newGain(board, self.Net, self.gamma)
            boardTensor = nn2048helper.makeTensor(board)
            currScore = getScore(board, self.Net, boardTensor)
            self.data.append([boardTensor, currScore*(1-self.alpha) + self.alpha*boardNewGain])
            board = nn2048helper.addNum(board)
            board, rewToNet, reward = makeMoveUnderPolicy(board, self.Net, self.gamma, self.eps)
            overallScore += reward
        print('Score: ', overallScore)

model = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

alpha = 1/pow(2,0.6)
eps = 0.3
batchSize = 128
gamma_1 = 0.5
batchInit = Batch(eps, batchSize)
xTrain, yTrain, = batchInit.run(None, alpha, 1-gamma_1, 1)
for j in range(20):
    lossCum = 0
    for k in range(len(xTrain)):
        print(j, k, len(xTrain))
        xTrainBatch, yTrainBatch = xTrain[k], yTrain[k]
        yPredBatch = model(xTrainBatch)
        loss = criterion(yPredBatch, yTrainBatch.unsqueeze(1))
        lossCum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        yPredBatch = model(xTrainBatch)
        loss = criterion(yPredBatch, yTrainBatch.unsqueeze(1))
    print(j, lossCum / len(xTrain))

for i in range(2, 111):
    batch = Batch(eps*pow(0.98, i-1), batchSize)
    xTrain, yTrain = batch.run(model, 1/pow(i+1, 0.6), 1-gamma_1*pow(0.98,i-1), i)
    lossCum = 0
    for k in range(len(xTrain)):
        #print(k, len(xTrain))
        xTrainBatch, yTrainBatch = xTrain[k], yTrain[k]
        yPredBatch = model(xTrainBatch)
        loss = criterion(yPredBatch, yTrainBatch.unsqueeze(1))
        lossCum += loss.item()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # yPredBatch = model(xTrainBatch)
        # loss = criterion(yPredBatch, yTrainBatch.unsqueeze(1))
    print(i, 'before training', lossCum/len(xTrain))
    for j in range(20):
        lossCum = 0
        for k in range(len(xTrain)):
            #print(k, len(xTrain))
            xTrainBatch, yTrainBatch = xTrain[k], yTrain[k]
            yPredBatch = model(xTrainBatch)
            loss = criterion(yPredBatch, yTrainBatch.unsqueeze(1))
            lossCum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            yPredBatch = model(xTrainBatch)
            loss = criterion(yPredBatch, yTrainBatch.unsqueeze(1))
        print(i, j, lossCum / len(xTrain))