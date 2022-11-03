#ifndef NN_H
#define NN_H
//#include "arduino.h"
#include "Arduino.h"
#include "math.h"
//#define uint8_t byte

class neural_network_api{
    public:
      float relu(float x);
      float leaky_relu(float x);
      int freeMemory();
      float MSE(float* x,float* y,byte s);
     
        
 
};
 class layer{
    private :
      // weights  private
      float ** wbuffers;
      float ** wbuffers2;
       // biases private
      float* wbiases;
     // float* momemtum ;
      float* wbiases2;
    public:
     float ** cw1; 
     byte actfn = 0; // activation function
     float* biases;
     float* acts  ; // get the acts of the layer
     float* gamma;
     int units = 1; // neuron cont
     int inputs;// inputs

     
     layer(int inputs , int neurons,byte inference=1){
       this -> gamma = new float[inputs]{}; 
       this -> cw1= new float * [neurons];
       if(inference==0){
         this -> wbuffers= new float * [neurons];
         this -> wbuffers2= new float * [neurons];
       }
       for (byte i = 0; i< neurons;i++){
          cw1[i] = new float [inputs];
          if(inference==0){
           wbuffers[i] = new float [inputs];
           wbuffers2[i] = new float [inputs];
          }
       }
     
       this ->inputs = inputs;
       this-> units =neurons;
       this -> acts = new float[neurons];
       this-> biases = new float [neurons];
       if(inference==0){
         this-> wbiases = new float [neurons];
       }
     };
     byte set_wb(byte neuron,float * new_weights, float bias){
        this -> cw1[neuron] = new_weights;
        this -> biases[neuron] = bias;
        return 1;
     };
     void init(byte o){
      this ->actfn = o;
     
      for (byte x = 0; x< units;x++){
         biases[x] = randomfloat()*2; //randomfloat(0.0,1.0);
          for (int y = 0; y< inputs;y++){
            cw1[x][y] =randomfloat()*2; 
           
          }
          
       } 
     };
     float ** cross_over(float ** p1){
        for (byte x = 0; x< units;x++){
         // biases[x] = randomfloat(); //randomfloat(0.0,1.0);
          for (int y = 0; y< inputs;y++){
            if(random(0,2)==1){
              wbuffers2[x][y] = p1[x][y]; 
              wbuffers[x][y] = cw1[x][y];
              
             
            }else{
               wbuffers[x][y] =  p1[x][y];
               wbuffers2[x][y] = cw1[x][y]; 
            }
          }
          
       } 
      this -> cw1 = wbuffers2;
      return wbuffers;
     
     }
     float * cross_overb(float * p1){
        for (byte x = 0; x< units;x++){
         // biases[x] = randomfloat(); //randomfloat(0.0,1.0);
   
            if(random(0,2)==1){
              wbiases[x] = p1[x]; 
              wbiases2[x] = biases[x];
              
             
            }else{
               wbiases[x] =  p1[x];
               wbiases2[x] = biases[x]; 
            }
          
          
       } 
      this -> biases = wbiases;
      return wbiases2;
     
     }
     float randomfloat()
      {
         float a = random(0,10000);
         return ((a/10000.0)-0.5) / 0.5;
      }
     #if !defined(As__No_Common_Serial_Support)  
     void mutate(float h = 0.7) {
        for(byte x=0;x<units;x++){
           biases[x] = biases[x]+(randomfloat()-randomfloat())*h;
           for(byte y=0;y<inputs;y++){
            
              cw1[x][y] = cw1[x][y] +(randomfloat()-randomfloat())*h;
           }
        }
     }
     void print_weights(){
       // Serial.print(" layer hash ");
     //   Serial.println(hash);
        Serial.println("weights: ");
        for(int x=0;x<units;x++){
            Serial.print("neuron " );
            Serial.print(x);
            Serial.print(" weights: ");
			Serial.print("{");
          for(int y=0;y<inputs;y++){
              
               
               Serial.print(this -> cw1[x][y],10); // neural nets are flexible, is not required a lot of decimals
			   Serial.print(",");	
		 }
		  Serial.print("}");
          Serial.print("  bias: ");
          Serial.print(this -> biases[x],10);
          Serial.println(" ");
         //  Serial.println(units);
        }
           
      
     };
     #endif
    //math functions 
    float relu(float x ){
          if (x > 0){
        return x;
        
        }else {
        return 0;
        }
    };
    float leaky_relu(float x){
      if (x > 0.0){
        return x;
      
      }else {
        return x*0.3;
         // using a alpha of 0.05
         }
    };
    float sigmoid(float x){
       return 1 / (1 +exp(-x));
    }
 
    //derivates
    float sigmoid_dev(float x){
       return (1 / (1 +exp(-x))) * (1-(1 / (1 +exp(-x)))) ;
    }
    float tanh_dev(float x)
    { 
      //  float x0 = exp(x);
        //float x1 = 1.0 / x0;
        return (1 - (tanh(x)*tanh(x)))+0.1;
    }
    float leaky_relu_dev(float x){
       if (x<0){
        return 0.3;
       }else{
        return 1 ;
       }
    }
    
    ///////////////////////////Functions ///////////////////////
    void backprop_out(float* expected,float* inputs_vals,float lr=0.01,float momemtum=0.001,float blr = 0.05){
      //preLgamma = new float[_numberOfInputs]{};  
      float bias_Delta = 1.0;
      float gamma_n = 0.0;
      
       for (byte y = 0; y <inputs; y++)
        {
          this->  gamma[y]=0;
        }
       for (byte x = 0; x < units; x++)
      {
        bias_Delta =1.0;

        

         if (this -> actfn== 4) {
             //
             
           gamma_n = ((2 / units) * (acts[x] - expected[x])) * sigmoid_dev(acts[x]);

           bias_Delta *=gamma_n;   
        }
        
        if (this -> actfn== 3) {
             //
             
           gamma_n = ((2 / units) * (acts[x] - expected[x])) * tanh_dev(acts[x]);

           bias_Delta *=gamma_n;   
        }
        if (this -> actfn== 2) {
          
           gamma_n = ((2 / units) * (acts[x] - expected[x]))* leaky_relu_dev(acts[x]);

           bias_Delta *=gamma_n;   
        }
        if (this -> actfn== 1) {
          
           gamma_n = ((2 / units) * (acts[x] - expected[x]));

           bias_Delta *=gamma_n;   
        }

        //////////////////////////////backbrop//////////////////
           //Serial.println("/////DEBUG/");
        for (int y = 0; y < inputs; y++)
        {
            if( not isnan((gamma_n * inputs_vals[y] * lr)  ) && not isinf((gamma_n * inputs_vals[y]) * lr)) {
              this-> cw1[x][y] -= (((gamma_n * inputs_vals[y]) - momemtum) * lr);
              this->  gamma[y] += gamma_n * cw1[x][y];
            }
            
          //  Serial.println(momemtum[y]);
      
        }
       if( not isnan(bias_Delta * lr) && not isinf(bias_Delta * lr)) {
          this -> biases[x] -= bias_Delta * blr;
        }
      }
    };


    void backprop_h(layer frontLayer,float* inputs_vals, float lr=0.01,float momemtum=0.001,float blr = 0.05){
//      preLgamma = new float[inputs]{};  
      float bias_Delta = 1.0;
      float gamma_n = 0.0;
      for (byte y = 0; y <inputs; y++)
        {
          this->  gamma[y]=0;
        }
       //Serial.println("/////DEBUG/");
     //  for (byte x = 0;x<units;x++){
     // Serial.println("////////*******************////////////");
       for (byte x = 0; x < units; x++)
      {
        bias_Delta = 1.0;
        if (frontLayer.actfn== 4) {
           gamma_n = frontLayer.gamma[x] * sigmoid_dev(acts[x]);

           bias_Delta *=gamma_n;   
        }
        
        if (frontLayer.actfn== 3) {
           gamma_n = frontLayer.gamma[x] * tanh_dev(acts[x]);

           bias_Delta *=gamma_n;   
        }
        if (frontLayer.actfn== 2) {
           gamma_n = frontLayer.gamma[x] * leaky_relu_dev(acts[x]);

           bias_Delta *=gamma_n;   
        }
         if (this -> actfn== 1) {
          
           gamma_n = frontLayer.gamma[x] ;

           bias_Delta *=gamma_n;   
        }
        

        //////////////////////////////backbrop//////////////////
         
        for (byte y = 0; y <inputs; y++)
        {
             
              if( not isnan((gamma_n * inputs_vals[y]) * lr) && not isinf((gamma_n * inputs_vals[y]) * lr)) {
              this-> cw1[x][y] -= ((((gamma_n) * inputs_vals[y]) -  momemtum ) * lr);
              this->  gamma[y] += gamma_n * cw1[x][y];
              // this ->momemtum[x] +=  ((gamma_n) * inputs_vals[y]) * 0.0001 ;
            }
           // Serial.print(  cw1[x][y]);
           //  Serial.print( "|");
        }
      //  Serial.print( " -> ");
      //  Serial.println( gamma[x]);
       
        if( not isnan(bias_Delta * blr) && not isinf(bias_Delta * blr)) {
          this->biases[x] -= bias_Delta * blr;
        }
    //    Serial.println( biases[i]);
      }
 //      Serial.println("////////*******************////////////");
      // Serial.println("/////END DEBUG/");
    };

    
    float* foward(float *x_input){
      //  acts = 0;
        float carry = 0.0;
       // for (byte n = 0;n<units;n++){
         //   acts[n] = 0.0; //act+ (x[n]*cw1[n]);
      //  }
         for (byte x = 0;x<units;x++){
             carry = 0.0;
           for (byte y = 0;y<inputs;y++){
              
             // for (byte z = 0;z<inputs;z++){
               carry = carry + (cw1[x][y] * x_input[y]);
              //}
             
              
           }
          //carry = carry+ ;
           carry +=biases[x];
           if(actfn == 0 ){
            acts[x] = carry;
           }
           if(actfn == 1 ){
            acts[x] = relu(carry);//+ biases[x];
           }
           if(actfn == 2){
            acts[x] = leaky_relu(carry);//+ biases[x];
           }
           if(actfn == 3){
            acts[x] = tanh(carry);//+ biases[x];
           }
           if(actfn==4){
            acts[x]=sigmoid(carry);
           }
             
        //  acts[x] = acts[x]+biases[x];
        //  Serial.println(acts[x]);
         }
       
        return acts;
        
    };
    
    //  void predecir(float* x);
      
      
       
 
};

#endif
