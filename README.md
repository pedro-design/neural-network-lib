# neural-network-lib
this is a proyect for implementing a mlp in an Arduino Mega

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
float test_array[2] = {8,-3}

my_layer.foward(test_array);

//to get an output of a layer there are 2 methods
//1 calling layer.foward(), returns the output array
float * layer_out = my_layer.foward(test_array);

//or using layer.acts
float * layer_out2 = my_layer.acts
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
```
