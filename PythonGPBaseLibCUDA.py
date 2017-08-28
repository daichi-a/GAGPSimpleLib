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

import pycuda.autoinit
import pycuda.driver
from pycuda.compiler import SourceModule


def symbolRegressionCUDA(programList, functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, kernelFunction, targetFunction, index):
    targetFunctionIndex = numpy.zeros(1, dtype=numpy.int32);
    if targetFunction == 'MSE':
        targetFunctionIndex[0] = 0
    elif targetFunction == 'MDL':
        targetFunctionIndex[0] = 1
    else:
        print('Wrong target function name is given!, using MSE')
        targetFunctionIndex[0] = 0
    timeLength = len(timeArray)

    for j in range(len(timeArray)):
        evaluatingProgramList = copy.copy(programList)
        variableDict['T'] = timeArray[j]
        replaceVariableWithStatic(evaluatingProgramList, variableDict)
        replaceNodeToIntLabelForCUDA(i, evaluatingProgramArrayInIntLabel, nodeArray, stackCountArray, evaluatingProgramList, functionDict, variableDict, stackCountDict, functionIndexInCUDADict)

        evaluateCUDA(evaluatingProgramList, functionDict, stackCountDict)
        fxArray[j] = evaluatingProgramList[0]

    subBuffer = numpy.zeros_like(fxArray)
    score = numpy.zeros(1, dtype=numpy.float32)
    lengthOfProgramList = numpy.zeros(1, dtype=numpy.int32)
    lengthOfProgramList[0] = len(programList)
    kernelFunction(pycuda.driver.Out(score), pycuda.driver.Out(subBuffer), pycuda.driver.In(targetFxArray), pycuda.driver.In(fxArray), pycuda.driver.In(targetFunctionIndex), pycuda.driver.In(lengthOfProgramList), block=(timeLength, 1, 1), grid=(1, 1))

    scoreArray[index] = score[0]
    return None


def evaluatePopulationSymbolRegressionCUDA(population, functionDict, variableDict, staticList, stackCountDict, timeArray, targetFxArray, scoreArray, targetFunction, functionIndexInCUDADict):
    #module = SourceModule(codecs.open('getSubThenPow.cuda', 'r').read())
    #kernelFunction = module.get_function('getSubThenPow')
    module = \
        SourceModule(codecs.open('evaluateSymbolRegression.cuda', 'r').read())
    evaluateFunction = module.get_function('evaluate_program_array')
e
    for i in range(len(population)):
        timeArray = timeArray.astype(numpy.float32)
        targetFxArray = targetFxArray.astype(numpy.float32)
        fxArray = numpy.array(targetFxArray)

        symbolRegressionCUDA(population[i], functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, kernelFunction, targetFunction, i, functionIndexInCUDADict)

def replaceVariableWithStatic(programList, variableDict):
    for index, aNode in enumerate(programList):
        if aNode in variableDict:
            programList[index] = variableDict[aNode]


def replaceNodeToIntLabelForCUDA(i, evaluatingProgramArrayInIntLabel, functionList, stackCountArray, evaluatingProgramList, functionDict, variableDict, stackCountDict, functionIndexInCUDADict):
    pass

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

def evaluteCUDA(programArray, functionDict, stackCountDict, functionIndexInCUDADict, evaluateSymbolRegressionKernelFunction):
    pass
