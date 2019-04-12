# -*- coding: utf-8 -*-
# for Python 3.7
# Simple Function Set Libarary for Genetic Programming in Python
# Daichi Ando @daichi_a daichi-a
from PythonGPBaseLib import replaceVariableWithStatic, evaluate, initializeProgramList

import random
import copy
import threading
import numpy
import codecs

from numba import jit

@jit
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

@jit
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

@jit
def symbolRegression(programList, functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, targetFunction, index, functionIndexInCUDADict):
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
    

@jit
def evaluatePopulationSymbolRegression(population, functionDict, variableDict, staticList, stackCountDict, timeArray, targetFxArray, scoreArray, targetFunction, functionIndexInCUDADict):
    for i in range(len(population)):
        timeArray = timeArray.astype(numpy.float32)
        targetFxArray = targetFxArray.astype(numpy.float32)
        fxArray = numpy.array(targetFxArray)

        symbolRegression(population[i], functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray,  targetFunction, i, functionIndexInCUDADict)
