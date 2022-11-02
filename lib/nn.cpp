#include "NN.h"
#include "Arduino.h"
//#ifdef __arm__
// should use uinstd.h to define sbrk but Due causes a conflict
//extern "C" char* sbrk(int incr);
//#else  // __ARM__
//extern char *__brkval;
//#endif  // __arm__



int neural_network_api::freeMemory() {
  char top;
#ifdef __arm__
  //return &top - reinterpret_cast<char*>(sbrk(0));
  return 0;
#elif defined(CORE_TEENSY) || (ARDUINO > 103 && ARDUINO != 151)
//  return &top - __brkval;
  return 0;
#else  // __arm__
  //return __brkval ? &top - __brkval : &top - __malloc_heap_start;
  return 0;
#endif  // __arm__
}


  
float neural_network_api::MSE(float*x,float*y,byte s){
  float sum =0.0;

  //do the sum
    for(byte i=0;i<s;i++){
      sum = sum +sqrt((x[i]-y[i])*(x[i]-y[i])) /s;
     
    }

   return sum;
  
};

float neural_network_api::relu(float x){
  if (x > 0){
    return x;
    
  }else {
    return 0;
  }
}

float neural_network_api::leaky_relu(float x){
  if (x > 0.0){
    return x;
      
  }else {
    return x*0.3;
    // using a alpha of 0.3
  }
}
