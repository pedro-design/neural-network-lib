# neural-network-lib
this is a proyect for implementing a mlp in a Arduino Mega

# To load the lib use:
it requires of math.h
```c++
#include <NN.h>
```

# mini guide
```c++
//firts load the lib
#include <NN.h>
//then create a layer object and add a name to it
// the layer object has 3 requiremets to create layer(input_size,units,low_mem_mode)
layer my_layer(2,4,1);

//now initialize the layer
// layer.init(activation function)
//this are the current activations supported:
// 0 : linear
// 1 : relu
// 2 : leaky_relu
// 3 : tanh
// 4 : sigmoid
my_layer.init(2);

//you can print the layer weights and biases if there is a serial port
my_layer.print_weights();

//to pass an input to the layer use layer.foward(array)
// there is no error indicator, so try that the size of the array and the input size of the layer are the same
float test_array[2] = {8,-3};

my_layer.foward(test_array);

//to get an output of a layer there are 2 methods
//1 calling layer.foward(), returns the output array
float * layer_out = my_layer.foward(test_array);

//or using layer.acts
float * layer_out2 = my_layer.acts;
//there are other public classes in the layer object like :
// layer.units returns the neurons in the layer
//layer.cw1 returns the 2D array of weights
//layer.biases retunrs a 1D array of biases
//layer.acfn returns the layer activation funtion
//layer.inputs returns the layer input size
//layer.gamma returns the gamma of the layer for backpropagation

//now the training part:
//for that use layer.backprop_out and layer.backprop_h
//backprop_out computes the derivate of the output layer
//backprop_h the derivate of the hiden layer
//the sintax is backprop_out(expected_output_arr,prev_layer_acts,lr,momentum)
//and for backprop_h(nex_layer,prev_layer_acts or the input_of_the_network,lr,momentum)
float expected[4] = {1,3,4,-3}

//this is the only layer so the prev_layer_acts are the input array
my_layer.backprop_out(expected,test_array,0.01,0.01);
//view the change of the layer
my_layer.print_weights();
```


# multiple layer model building and training
```c++

#include <NN.h>


//then create the layers
// the layer object has 3 requiremets to create layer(input_size,units,low_mem_mode)
layer input(2,4,1);
layer hiden_layer_1(input.units,4,1); // using the prev layer units for more easy model building
layer out_layer(hiden_layer_1.units,1,1);

//now initialize each layer

void setup(){
   Serial.begin(19200); // start the serial port
// layer.init(activation function)
//this are the current activations supported:
// 0 : linear
// 1 : relu
// 2 : leaky_relu
// 3 : tanh
// 4 : sigmoid
input.init(2); // relu
hiden_layer_1.init(3); // leaky_relu
out_layer.init(4);// sigmoid
//view the layers weights
input.print_weights();
hiden_layer_1.print_weights();
out_layer.print_weights();
// the data for this example will be the AND problem
float x_train[4][2] = {{0,0},{1,0},{0,1},{1,1}};
float y_train[4][1] = {{0},{0},{0},{1}};


 float loss = 0.0;

 for (int g=0;g<1000;g++){ // run 500 iterations
    loss = 0.0;
    for (int t=0;t<4;t++){
      //pass foward the network
      input.foward(x_train[t]);
      hiden_layer_1.foward(input.acts);
      float * outs = out_layer.foward(hiden_layer_1.acts);
      
      //compute backpropagation
      //backprop_out computes the derivate of the output layer
      //backprop_h the derivate of the hiden layer
      //the sintax is backprop_out(expected_output_arr,prev_layer_acts,lr,momentum)
      //and for backprop_h(nex_layer,prev_layer_acts or the input_of_the_network,lr,momentum)
      
       out_layer.backprop_out(y_train[t],hiden_layer_1.acts,0.01,0.01);
       hiden_layer_1.backprop_h(out_layer,input.acts,0.01,0.01);
       input.backprop_h(hiden_layer_1,x_train[t],0.01,0.01);
       loss += sqrt( (outs[0]-y_train[t][0])*(outs[0]-y_train[t][0]) ) ; // RMSE loss
        
     }
      loss = loss /4;
      Serial.print(" loss: "); // print the loss
      Serial.println(loss,5);
    }
    //print new weights
  input.print_weights();
  hiden_layer_1.print_weights();
  out_layer.print_weights();
  //now view the network predictions
   for (int t=0;t<4;t++){
      input.foward(x_train[t]);
      hiden_layer_1.foward(input.acts);
      float * outs = out_layer.foward(hiden_layer_1.acts);
      Serial.print(" inputs: "); // print the loss
      Serial.print(x_train[t][0]);
       Serial.print(" | ");
      Serial.print(x_train[t][1]);
      Serial.print(" out: ");
      Serial.print(outs[0],5);
      Serial.print(" expected: ");
      Serial.println(y_train[t][0],2);
   }
}
//this is the end of this tutorial

void loop() {
  // nothing here 

}


```
