# -*- coding: utf-8 -*-
# for Python 3.6
# Simple Function Set Libarary for Genetic Programming in Python
# Daichi Ando @daichi_a daichi-a

import random
import copy
import threading
import numpy
import time

def initializeProgramList(functionDict, variableDict, staticList, stackCountDict, limitOfMinSC):
    currentStackCount = 0
    programList = []
    programList.append(random.choice(list(functionDict.keys())))
    currentStackCount += stackCountDict[programList[-1]] if programList[-1] in stackCountDict else 1
    while(currentStackCount < 1):
        randomValue = random.random()
        if currentStackCount < limitOfMinSC:
            programList.append(random.choice(staticList)) if randomValue < 0.8 else programList.append(random.choice(list(variableDict.keys())))
        elif randomValue < 0.3:
            programList.append(random.choice(list(functionDict.keys())))
        elif randomValue < 0.6:
            programList.append(random.choice(list(variableDict.keys())))
        else:
            programList.append(random.choice(staticList))
        currentStackCount += stackCountDict[programList[-1]] if programList[-1] in stackCountDict else 1
    return programList

def initializePopulation(populationSize, functionDict, variableDict, staticList, stackCountDict, limitOfLowerSC):
    population = []
    for i in range(populationSize):
        population.append(initializeProgramList(functionDict, variableDict, staticList, stackCountDict, limitOfLowerSC))
    return population


def replaceVariableWithStatic(programList, variableDict):
    for index, aNode in enumerate(programList):
        if aNode in variableDict:
            programList[index] = variableDict[aNode]
        
def evaluate(programList, functionDict, stackCountDict):
    while(len(programList) > 1):
        #Detect a last function node to evaluate first
        indexOfLastFunctionNode = -1;
        for index, aNode in enumerate(reversed(programList)):
            if type(aNode) is str:
                indexOfLastFunctionNode = len(programList) - index -1
                break
        #Extract a subtree to evaluate
        evaluatingSubTree = []
        subTreeStackCount = 0
        while(subTreeStackCount < 1):
            evaluatingSubTree.append(programList.pop(indexOfLastFunctionNode))
            subTreeStackCount += stackCountDict.get(evaluatingSubTree[-1], 1)
        #Evaluate the subtree then replace subtree with the evaluated result
        evaluatedValue = functionDict[evaluatingSubTree[0]](evaluatingSubTree)
        programList.insert(indexOfLastFunctionNode, evaluatedValue)


def symbolRegression(programList, functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, targetFunction, index):
    # Argument: targetFunction 'MSE' or 'MDL'
    # In Genetic Programming GP Symbol Regression, MDL formula as follows:
    # MDL = (MSE * lengthOfSample) + lengthOfTree * log(lengthOfSample)
    # is used.
    # https://www.jstage.jst.go.jp/article/sicejl1962/39/5/39_5_352/_pdf
    

    sumOfScore = 0.0
    for j in range(len(timeArray)):
        evaluatingProgramList = copy.copy(programList)
        variableDict['T'] = timeArray[j]
        replaceVariableWithStatic(evaluatingProgramList, variableDict)
        evaluate(evaluatingProgramList, functionDict, stackCountDict)
        fxArray[j] = evaluatingProgramList[0]

    if targetFunction == 'MSE':
        scoreMSE = numpy.mean(numpy.power((targetFxArray - fxArray), 2))
        scoreArray[index] = scoreMSE
    elif targetFunction == 'MDL':
        scoreMSE = numpy.mean(numpy.power((targetFxArray - fxArray), 2))
        scoreMDL = scoreMSE * len(timeArray) + len(programList) * numpy.log(len(timeArray))
        scoreArray[index] = scoreMDL
    else:
        print('Wrong Target Function is given!, using MSE')
        scoreArray[index] = numpy.mean(numpy.power((targetFxArray - fxArray), 2))
    
    return None


def evaluatePopulationSymbolRegression(population, functionDict, variableDict, staticList, stackCountDict, timeArray, targetFxArray, scoreArray, targetFunction):
    for i in range(len(population)):
        fxArray = numpy.array(targetFxArray)

        #If you want use multi-threading, use these 2 lines:
        #th_me = threading.Thread(target=symbolRegression, name="th_me", args=(population[i], functionDict, variableDict, staticList, fxArray, timeArray, targetFxArray, rssArray, i))
        #th_me.start()

        #However, Python codes without standard library
        #cannot be deal with multi-threading effectively.
        #Thus the process written in single as follows:
        symbolRegression(population[i], functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, targetFunction, i)


def scorePopulation(population, scoreArray, scoreOrder, operation, generation, filename):
    #scoreOrder: 'UP': lower score is better, 'DONW': higher socre is better
    #scoreOrder: 'UP': スコアが小さい方が高評価, 'DOWN': スコアが大きい方が高評価
    if operation == 'log':
        averageScore = numpy.mean(scoreArray)
        if scoreOrder == 'DOWN':
            bestScore = numpy.max(scoreArray)
            scoreSortedIndexNumpyArray = numpy.argsort(scoreArray)[::-1]
        else:
            bestScore = numpy.min(scoreArray)
            scoreSortedIndexNumpyArray = numpy.argsort(scoreArray)

        bestScoreIndex = scoreSortedIndexNumpyArray[0]
        bestIndividual = population[bestScoreIndex]

        # Write to stdout
        log_string = '{0},{1},{2},{3},{4}\n'.format(generation, bestScore, averageScore, bestIndividual, time.ctime(time.time()))
        print(log_string)

        # Write to file same information
        if generation == 0:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a')
        try:
            file.write(log_string)
        finally:
            file.close()


def crossoverProgramList(programList1, programList2, stackCountDict):
    crossoverPoint1 = random.randrange(len(programList1))
    subTree1 = []
    subTreeStackCount1 = 0
    while(subTreeStackCount1 < 1):
        subTree1.append(programList1.pop(crossoverPoint1))
        subTreeStackCount1 += stackCountDict.get(subTree1[-1], 1)

    crossoverPoint2 = random.randrange(len(programList2))
    subTree2 = []
    subTreeStackCount2 = 0
    while(subTreeStackCount2 < 1):
        subTree2.append(programList2.pop(crossoverPoint2))
        subTreeStackCount2 += stackCountDict.get(subTree2[-1], 1)
    
    while(len(subTree2) > 0):
        programList1.insert(crossoverPoint1, subTree2.pop())
    while(len(subTree1) > 0):
        programList2.insert(crossoverPoint2, subTree1.pop())
    

def crossoverPopulation(population, newPopulation, scoreArray, tournamentUnitSize, scoreOrder, stackCountDict, numOfElite):
    #scoreOrder: 'UP': ascending order: lower score is better,
    # 'DOWN' descending order: hight score is better.

    #NumOfElite: number of individual to store into next generation by elite strategy

    #scoreOrder: 'UP': スコアが小さい方が高評価, 'DOWN': スコアが大きい方が高評価
    #numOfElite: エリート戦略ととる場合の次代へ残すエリートの数．0だとやらない．

    if numOfElite > 0:
        if scoreOrder == 'DOWN':
            scoreIndexSortedNumpyArray = numpy.argsort(scoreArray)[::-1]
        else:
            scoreIndexSortedNumpyArray = numpy.argsort(scoreArray)
        for i in range(numOfElite):
            newPopulation.append(copy.copy(population[scoreIndexSortedNumpyArray[i]]))
    
    while len(newPopulation) < len(population):
        parents = []
        for i in range(2):
            tournamentUnit = []
            tournamentUnitScoreArray = []
            while len(tournamentUnit) < tournamentUnitSize:
                tournamentUnit.append(random.choice(population))
                tournamentUnitScoreArray.append(scoreArray[population.index(tournamentUnit[-1])])
            if scoreOrder == 'DOWN':
                parents.append(tournamentUnit[tournamentUnitScoreArray.index(max(tournamentUnitScoreArray))])
            else:
                parents.append(tournamentUnit[tournamentUnitScoreArray.index(min(tournamentUnitScoreArray))])
        offspring1 = copy.copy(parents[0])
        offspring2 = copy.copy(parents[1])
        crossoverProgramList(offspring1, offspring2, stackCountDict)
        newPopulation.append(offspring1)
        newPopulation.append(offspring2)

def mutateProgramList(programList, functionDict, variableDict, staticList, stackCountDict, mutationRate):
    currentIndex = 0
    while(currentIndex < len(programList) -1):
        subTreeLength = 1
        if random.random() < mutationRate:
            replacingSubTree = initializeProgramList(functionDict, variableDict, staticList, stackCountDict, -10)
            subTreeLength = len(replacingSubTree)
            removingSubTreeStackCount = 0
            while(removingSubTreeStackCount < 1):
                currentNode = programList.pop(currentIndex)
                removingSubTreeStackCount += stackCountDict.get(currentNode) if currentNode in stackCountDict else 1
            while(len(replacingSubTree) > 0):
                programList.insert(currentIndex, replacingSubTree.pop())
        currentIndex += subTreeLength

def mutatePopulation(population, functionDict, variableDict, staticList, stackCountDict, mutationRate, numOfElite):
    threadList = []
    for index, programList in enumerate(population):
        #These 4 lines are for multi-threading:
        #th_me = threading.Thread(target=mutateProgramList, name="th_me", args=(programList, functionDict, variableDict, staticList, stackCountDict, mutationRate))
        #threadList.append(th_me)
        #th_me.start()
    #for aThread in threadList:
        #aThread.join()

        #On the contrary, this one line is for single-threading:
        if index < numOfElite:
            pass
        else:
            mutateProgramList(programList, functionDict, variableDict, staticList, stackCountDict, mutationRate)
