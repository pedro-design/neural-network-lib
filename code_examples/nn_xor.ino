//#include <Servo.h>

#include "NN.h"

float s=0;
unsigned long last_time, now_time;
long count;
neural_network_api nn;

layer c1_input_1(2,4,1);// layer(input_size,units,mode,inference =1 or o)
layer c1_hiden_1(c1_input_1.units,4,1);
layer c1_hiden_2(c1_hiden_1.units,4,1);
layer c1_out(c1_hiden_2.units,2,1);//neural network layers

float x_train[4][2] = {{0,1},{1,1},{0,0},{1,0}};
float y_train[4][2] ={{1,0},{0,1},{0,1},{1,0}};


// ga paramatters
// [size][neurons][input_size]




//float layer1w[2][2] = {{-7.565603648566 , 7.7076089635465} , {4.0793788457077 , -5.884165060772}}; // to set weights create an array x[neurons_of_layers][inputs_of_layers]
//float layer1b[2] = {-5.4436171763643 , 0.73903108395501};

//float layer2w[2][2] ={{0.092872577357259 , -1.3577316830489} , {-1.6912347061153 , -2.5793021663442}};
//float layer2b[2] = {-0.55129175387214 , 4.5411099863426};


//float layer3w[1][2]={{-0.95306233498775 , -2.6579067340226}};
//float layer3b[1] = {1.5434943630096};//0.5974

//Servo myservo;

    
void setup() {
 
 //  myservo.attach(9);
  // put your setup code here, to run once:
//randomSeed(5);
  last_time = micros();
 Serial.begin(19200);
 // initialize the weights
  
  Serial.println("start");
//  Serial.println(layer1w[0][0]);
  c1_input_1.init(2);// initialize the weights of the layer
  c1_hiden_1.init(2);// init(act_function)
   c1_hiden_2.init(2);// init(act_function)
  c1_out.init(4);

  c1_input_1.print_weights();//print weights
  c1_hiden_1.print_weights();//print weights
  c1_out.print_weights();
 // Serial.println(nn.freeMemory());

//test

  float loss;
  float sum = 0.0;
  float t_loss = 1.0;
  for (int g=0;g<5000;g++){
    //foward
  
   
     sum = 0.0;
    for (int t=0;t<4;t++){
      c1_input_1.foward(x_train[t]);
      c1_hiden_1.foward(c1_input_1.acts);
      c1_hiden_2.foward(c1_hiden_1.acts);
      float * outs = c1_out.foward(c1_hiden_2.acts);
      //calculate loss 
          
      
      
// 
            loss = (outs[0]-y_train[t][0]) ;
             loss += (outs[1]-y_train[t][1]) ;
//          Serial.print(x_train[t][0]);
          Serial.print("|");
//             Serial.print(x_train[t][1]);
//          Serial.print(" -out ");
//          Serial.print(outs[0]);
//           Serial.print(" -exp ");
//           
//           Serial.print(y_train[t][0]);
//             Serial.print(" index: ");
//          Serial.println(t);
         sum +=sqrt(loss*loss);
        

          
      
      if (t_loss> 0.05 && t_loss< 10) {
          c1_out.backprop_out(y_train[t],c1_hiden_1.acts,0.01,0.01);
          c1_hiden_2.backprop_h(c1_out,c1_hiden_1.acts,0.01,0.01);
          c1_hiden_1.backprop_h(c1_hiden_2,c1_input_1.acts,0.01,0.01);
          c1_input_1.backprop_h(c1_hiden_1,x_train[t],0.01,0.01);
          }else{
            break;
          }
     //// backprop
     
     
    }
     if (t_loss> 0.05 && t_loss< 10) {
        Serial.print(" loss: ");
        Serial.println(sum/4,5);
        t_loss = sum/4;
     }else{
      break;
     }
  
  }
  c1_input_1.print_weights();//print weights
   c1_hiden_2.print_weights();//print weights
  c1_hiden_1.print_weights();//print weights
 
  c1_out.print_weights();
  for (int t=0;t<4;t++){
         c1_input_1.foward(x_train[t]);
         c1_hiden_1.foward(c1_input_1.acts);
         c1_hiden_2.foward(c1_hiden_1.acts);
         float * outs = c1_out.foward(c1_hiden_2.acts);
         loss = (outs[0]-y_train[t][0]) ;
         Serial.print(x_train[t][0]);
         Serial.print("|");
         Serial.print(x_train[t][1]);
         Serial.print(" -out ");
         Serial.print(outs[0]);
         Serial.print(" | ");
         Serial.print(outs[1]);
         Serial.print(" -exp ");
         Serial.print(y_train[t][0]);
         Serial.print(" index: ");
         Serial.println(t);
  }
 

}

void loop() {
 
 
  
}
