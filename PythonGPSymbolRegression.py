# -*- coding: utf-8 -*-
# for Python 3.7
# This codes is a example of using the my Genetic Programming Library.
# In this codes, Symbolic Regression with normal GA Scenario.
# Daichi Ando @daichi_a daichi-a

import PythonGPBaseLib #My GP Library

import random
import copy
import math
import numpy

#Set a seed value you want -----------------------------------------------------
random_seed = 4
random.seed(random_seed) 

#Set GP Parameters -------------------------------------------------------------

#Number of Individual in a population
populationSize = 1000
#Number of Loop Generations
generationLimit = 1000

#Mutation Rate
mutationRate = 0.05

# Number of Elite Individual to store next generation directory
numOfElite = 4

#Koza's style limitaiton of tree depth
limitationDepthOfInitialCodes = -17 

#Target Function
targetFunction = 'MDL'

log_file_name = '{0}{1}{2}{3}{4}{5}{6}{7}'.format('GPMainSymbolRegressionSeed',random_seed,'Size',populationSize, 'Gen', generationLimit, targetFunction, '.log')

#Initialize GP Node Set --------------------------------------------------------
#Function Nodes
functionDict = {}
stackCountDict = {}

#In this example, 7 function nodes, "+, -, *, /, IfLessThenElse, cos, sin" are implemented with lambda.
functionDict['add'] = (lambda programList: programList[1] + programList[2])
stackCountDict['add'] = -1

functionDict['sub'] = (lambda programList: programList[1] - programList[2])
stackCountDict['sub'] = -1

functionDict['mul'] = (lambda programList: programList[1] * programList[2])
stackCountDict['mul'] = -1

functionDict['div'] = (lambda programList: programList[1] / programList[2] if programList[2] != 0 else 1)
stackCountDict['div'] = -1

functionDict['IfLessThanElse'] = (lambda programList: programList[3] if programList[1] > programList[2] else programList[4])
stackCountDict['IfLessThanElse'] = -3

functionDict['cos'] = (lambda programList: math.cos(programList[1]))
stackCountDict['cos'] = 0

functionDict['sin'] = (lambda programList: math.sin(programList[1]))
stackCountDict['sin'] = 0

#Variable Value Nodes
variableDict = {}
#Variable T means horizontal axis of the symbol regression "Time"
variableDict['T'] = 1.0
stackCountDict['T'] = 1

#Static Value Nodes
#10 static value nodes are generated with a expression as follows:
staticList = [round((0.1 * f + 0.1), 1) for f in range(10)]

#Target Function
# Generating target function curve with a expression
timeArray = numpy.array([round((j * 0.01 + 0.01), 2) for j in numpy.arange(1000)])
targetFxArray = numpy.array(list(map(lambda t: 3 * t ** 3 + -2 * t ** 2 + 6 * t + 4, timeArray)))



#Now Start the GP in Normal GA Scenario ----------------------------------------
population = PythonGPBaseLib.initializePopulation(populationSize, functionDict, variableDict, staticList, stackCountDict, limitationDepthOfInitialCodes)

for i in range(generationLimit):
    scoreArray = numpy.zeros(populationSize, dtype=float)
    PythonGPBaseLib.evaluatePopulationSymbolRegression(population, functionDict, variableDict, staticList, stackCountDict, timeArray, targetFxArray, scoreArray, targetFunction)
    PythonGPBaseLib.scorePopulation(population, scoreArray, 'UP', 'log', i, log_file_name)
    newPopulation = []
    PythonGPBaseLib.crossoverPopulation(population, newPopulation, scoreArray, 5, 'UP', stackCountDict, numOfElite)
    PythonGPBaseLib.mutatePopulation(newPopulation, functionDict, variableDict, staticList, stackCountDict, mutationRate, numOfElite)
    population = newPopulation
