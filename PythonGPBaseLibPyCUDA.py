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

import pycuda.autoinit
import pycuda.driver
from pycuda.compiler import SourceModule


def symbolRegression(programList, functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, kernelFunction, targetFunction, index):
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


def evaluatePopulationSymbolRegression(population, functionDict, variableDict, staticList, stackCountDict, timeArray, targetFxArray, scoreArray, targetFunction, functionIndexInCUDADict):
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

        symbolRegression(population[i], functionDict, variableDict, staticList, stackCountDict, fxArray, timeArray, targetFxArray, scoreArray, kernelFunction, targetFunction, i, functionIndexInCUDADict)
