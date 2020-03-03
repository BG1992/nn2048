import torch
import numpy as np
import random
from math import log2

def isUpPossible(board):
    for col in range(4):
        isZero = False
        for row in range(4):
            if board[row, col] == 0:
                isZero = True
            else:
                if isZero:
                    return True
            if row > 0:
                if board[row, col] == board[row-1, col]:
                    return True
    return False

def isDownPossible(board):
    for col in range(4):
        isZero = False
        for row in range(3, -1, -1):
            if board[row, col] == 0:
                isZero = True
            else:
                if isZero:
                    return True
            if row < 3:
                if board[row, col] == board[row+1, col]:
                    return True
    return False

def isLeftPossible(board):
    for row in range(4):
        isZero = False
        for col in range(4):
            if board[row, col] == 0:
                isZero = True
            else:
                if isZero:
                    return True
            if col > 0:
                if board[row, col] == board[row, col-1]:
                    return True
    return False

def isRightPossible(board):
    for row in range(4):
        isZero = False
        for col in range(3, -1, -1):
            if board[row, col] == 0:
                isZero = True
            else:
                if isZero:
                    return True
            if col < 3:
                if board[row, col] == board[row, col+1]:
                    return True
    return False

def checkMoves(board):
    moves = []
    if board is None: return []
    if isUpPossible(board): moves.append('U')
    if isDownPossible(board): moves.append('D')
    if isLeftPossible(board): moves.append('L')
    if isRightPossible(board): moves.append('R')
    return moves

def makeMove(board, direction):
    newBoard = board.copy()
    reward = 0
    if direction == 'U':
        for col in range(4):
            nums = []
            for row in range(4):
                if board[row, col] != 0:
                    nums.append(board[row, col])
            indCum = 0
            i = 1
            while i < len(nums):
                if nums[indCum] == nums[i]:
                    nums[indCum] += nums[i]
                    reward += nums[i]*2
                    nums.pop(i)
                indCum += 1
                i = indCum + 1
            if len(nums) > 0:
                nums.extend([0]*(4-len(nums)))
                newBoard[:,col] = nums
    elif direction == 'D':
        for col in range(4):
            nums = []
            for row in range(3, -1, -1):
                if board[row, col] != 0:
                    nums.append(board[row, col])
            indCum = 0
            i = 1
            while i < len(nums):
                if nums[indCum] == nums[i]:
                    nums[indCum] += nums[i]
                    reward += nums[i]*2
                    nums.pop(i)
                indCum += 1
                i = indCum + 1
            if len(nums) > 0:
                nums.extend([0]*(4-len(nums)))
                newBoard[:,col] = nums[::-1]
    elif direction == 'L':
        for row in range(4):
            nums = []
            for col in range(4):
                if board[row, col] != 0:
                    nums.append(board[row, col])
            indCum = 0
            i = 1
            while i < len(nums):
                if nums[indCum] == nums[i]:
                    nums[indCum] += nums[i]
                    reward += nums[i]*2
                    nums.pop(i)
                indCum += 1
                i = indCum + 1
            if len(nums) > 0:
                nums.extend([0]*(4-len(nums)))
                newBoard[row,:] = nums
    elif direction == 'R':
        for row in range(4):
            nums = []
            for col in range(3, -1, -1):
                if board[row, col] != 0:
                    nums.append(board[row, col])
            indCum = 0
            i = 1
            while i < len(nums):
                if nums[indCum] == nums[i]:
                    nums[indCum] += nums[i]
                    reward += nums[i]*2
                    nums.pop(i)
                indCum += 1
                i = indCum + 1
            if len(nums) > 0:
                nums.extend([0]*(4-len(nums)))
                newBoard[row,:] = nums[::-1]
    rewToNet = log2(reward+1)/2
    return newBoard, rewToNet, reward

def makeTensor(board):
    boardTensor = torch.zeros(18, 4, 4)
    for row in range(4):
        for col in range(4):
            if board[row, col] > 0:
                channel = int(log2(board[row, col]))-1
                boardTensor[channel][row][col] = 1
    return boardTensor

def initBoard():
    board = np.zeros([4,4]).astype(int)
    num1 = random.randint(0, 15)
    num2 = random.randint(0, 15)
    while num2 == num1:
        num2 = random.randint(0, 15)
    row1, col1 = num1 // 4, num1 % 4
    row2, col2 = num2 // 4, num2 % 4
    if random.random() <= 0.9:
        board[row1, col1] = 2
    else:
        board[row1, col1] = 4
    if random.random() <= 0.9:
        board[row2, col2] = 2
    else:
        board[row2, col2] = 4
    return board

def addNum(board):
    newBoard = board.copy()
    emptyCells = []
    for row in range(4):
        for col in range(4):
            if board[row,col] == 0:
                emptyCells.append([row, col])
    if len(emptyCells) > 0:
        cellToFill = random.randint(0, len(emptyCells) - 1)
        row, col = emptyCells[cellToFill]
        if random.random() <= 0.9:
            newBoard[row,col] = 2
        else:
            newBoard[row,col] = 4
        return newBoard
    return None

def countEmptyCells(board):
    tempCt = 0
    for row in range(4):
        for col in range(4):
            if board[row, col] == 0:
                tempCt += 1
    return tempCt

def addSomePossibleNums(board):
    emptyCells = []
    for row in range(4):
        for col in range(4):
            if board[row,col] == 0:
                emptyCells.append([row, col])
    if len(emptyCells) > 0:
        newBoards = []
        cellsToFill = random.choice(emptyCells, len(emptyCells)//5 +1)
        for cell in cellsToFill:
            newBoard = board.copy()
            newBoard[cell[0], cell[1]] = 2
            newBoards.append([newBoard, 0.9])
            newBoard = board.copy()
            newBoard[cell[0], cell[1]] = 4
            newBoards.append([newBoard, 0.1])
        return newBoards

def addAllPossibleNums(board):
    emptyCells = []
    for row in range(4):
        for col in range(4):
            if board[row,col] == 0:
                emptyCells.append([row, col])
    if len(emptyCells) > 0:
        newBoards = []
        for cell in emptyCells:
            newBoard = board.copy()
            newBoard[cell[0], cell[1]] = 2
            newBoards.append([newBoard, 0.9])
            newBoard = board.copy()
            newBoard[cell[0], cell[1]] = 4
            newBoards.append([newBoard, 0.1])
        return newBoards

def randomBoard(emptyCellsNum):
    board = np.zeros([4,4]).astype(int)
    emptyCells = []
    for i in range(4):
        for j in range(4):
            emptyCells.append([i,j])
    for em in range(16-emptyCellsNum):
        cellToFill = random.choice(emptyCells)
        row, col = cellToFill[0], cellToFill[1]
        board[row, col] = pow(2, random.randint(1, 12))
        emptyCells.remove(cellToFill)
    return board
