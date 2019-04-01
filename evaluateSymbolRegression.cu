__global__ void evaluateSymbolRegression(float* resultScore, float* result, float* programArray, float* evaluateBuffer, int* stackCountArray, int* programLength, int *maxProgramLengthFromMain, int *targetFunction, float* targetValueArray){
  // allocate buffer for processing
  const unsigned int maxProgramLength = maxProgramLengthFromMain[0];
  // extern __shared__ float copyBuffer[];

  const int index = threadIdx.x;

  int posCounter = programLength[index] - 1;

  // Copy a program array(individual) into buffer for processing
  while(posCounter >= 0){
    evaluateBuffer[index*maxProgramLength + posCounter] = 
      programArray[index*maxProgramLength + posCounter];
    posCounter--;
  }
  
  unsigned int currentProgramLength = programLength[index];
  while(currentProgramLength > 1){
    
    unsigned int currentProgramNode = 0;
    for(int i=currentProgramLength-1; i>=0; i--){
      // Remainder with 1.0 equals 0.0 (means integer),
      // stack count is less than zero,
      // detected as function node
      if(fmod((double)evaluateBuffer[index*maxProgramLength + i], 1.0) == 0.0 && 
	 (int)evaluateBuffer[index*maxProgramLength + i] > 65535 &&
	 stackCountArray[(int)evaluateBuffer[index*maxProgramLength + i] - 65535] <= 0){
	// Search Last Function Node
	currentProgramNode = i;
	break;
      }
    }

    int currentStackCount = 
      stackCountArray[(int)evaluateBuffer[index*maxProgramLength + currentProgramNode] -65535];
    unsigned int subTreeRange = 1;

    while(currentStackCount < 1){
      currentStackCount += 1;
      subTreeRange++;
    }

    float resultFromFunctionNode = 0.0;
    unsigned int functionLabel = 
      (int)evaluateBuffer[index*maxProgramLength + currentProgramNode] - 65535;

    switch(functionLabel){
    case 0: //add
      resultFromFunctionNode = 
	evaluateBuffer[index*maxProgramLength + currentProgramNode +1] 
	+ 
	evaluateBuffer[index*maxProgramLength + currentProgramNode +2];
      break;
    case 1: //sub
      resultFromFunctionNode = 
	evaluateBuffer[index*maxProgramLength + currentProgramNode +1] 
	- 
	evaluateBuffer[index*maxProgramLength + currentProgramNode +2];
      break;
    case 2: //mul
      resultFromFunctionNode = 
	evaluateBuffer[index*maxProgramLength + currentProgramNode +1] 
	*
	evaluateBuffer[index*maxProgramLength + currentProgramNode +2];
      break;
    case 3: //div
      if(evaluateBuffer[index*maxProgramLength + currentProgramNode + 1] != 0.0 &&
	 evaluateBuffer[index*maxProgramLength + currentProgramNode + 2] != 0.0){
	resultFromFunctionNode = 
	  evaluateBuffer[index*maxProgramLength + currentProgramNode +1] 
	  /
	  evaluateBuffer[index*maxProgramLength + currentProgramNode +2];
      }
      else
	resultFromFunctionNode = 1.0;
      break;
    case 4: //IfLessThenElse
      if(evaluateBuffer[index*maxProgramLength + currentProgramNode +1] > 
	 evaluateBuffer[index*maxProgramLength + currentProgramNode +2])
	resultFromFunctionNode = 
	  evaluateBuffer[index*maxProgramLength + currentProgramNode + 3];
      else
	resultFromFunctionNode =
	  evaluateBuffer[index*maxProgramLength + currentProgramNode + 4];
      break;
    case 5: //cos
      resultFromFunctionNode = 
	cos(evaluateBuffer[index*maxProgramLength + currentProgramNode + 1]);
      break;
    case 6: //sin
      resultFromFunctionNode = 
	sin(evaluateBuffer[index*maxProgramLength + currentProgramNode + 1]);
      break;
    default:
      resultFromFunctionNode = 0.0;
      break;
    }
    
    // Replace current function node with result of calc subtree
    evaluateBuffer[index*maxProgramLength + currentProgramNode] = 
      resultFromFunctionNode;
    // Copy node array after current subtree.
    for(int i=0; i<subTreeRange; i++)
      evaluateBuffer[index*maxProgramLength + currentProgramNode + 1 + i] = 
	evaluateBuffer[index*maxProgramLength + currentProgramNode + subTreeRange + i];
    // new currentProgramLength
    currentProgramLength = currentProgramLength - subTreeRange + 1;
    // Fill zero after currentProgramLength
    for(int i=index*maxProgramLength + currentProgramLength; 
	i<(index+1)*maxProgramLength;
	i++)
      evaluateBuffer[i] = 0.0;
  }
  result[index] = (float)evaluateBuffer[index*maxProgramLength] - targetValueArray[index] * evaluateBuffer[index*maxProgramLength] - targetValueArray[index];
  
  __syncthreads(); // Waiting Finish All Threads
  
 resultScore[0] = 0.0;

  if(index == 0){ // If thread ID == 0 then calc score from RSS array
      float sum = 0;
      for(unsigned int j=0; j<blockDim.x; j++){
          sum += result[j];
      }
      if(targetFunction[0] == 0){ //MSE: Mean Square Error
	resultScore[0] = (sum / blockDim.x);
      }
      else if(targetFunction[0] == 1){ //MDL: Minimum Description Length
	resultScore[0] = (sum / blockDim.x) + (programLength[0] * logf(blockDim.x));
      }
      else{
	resultScore[0] = 0.0;
      }
  }
}
