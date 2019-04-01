# -*- coding: utf-8 -*-
# for Python 3.6
# Simple Function Set Libarary for Genetic Programming in Python
# Daichi Ando @daichi_a daichi-a
import PythonGPBaseLib

import random
import copy
import threading
import numpy
import codecs

from numba import jit

@jit
def symbolRegressionNumba(programList, functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, targetFunction, index):
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
def evaluatePopulationSymbolRegressionNumba(population, functionDict, variableDict, staticList, stackCountDict, timeArray, targetFxArray, scoreArray, targetFunction):
    for i in range(len(population)):
        fxArray = numpy.array(targetFxArray)
        symbolRegressionNumba(population[i], functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, targetFunction, i)

        
@jit        
def replaceVariableWithStatic(programList, variableDict):
    for index, aNode in enumerate(programList):
        if aNode in variableDict:
            programList[index] = variableDict[aNode]

@jit
def replaceNodeToIntLabelForNumba(i, evaluatingProgramArrayInIntLabel, functionList, stackCountArray, evaluatingProgramList, functionDict, variableDict, stackCountDict, functionIndexInCUDADict):
    pass

@jit
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

def evaluteNumba(programArray, functionDict, stackCountDict, functionIndexInCUDADict, evaluateSymbolRegressionKernelFunction):
    pass
