#ifndef SNN_H
#define SNN_H
//TODO test the learning function
//#include <Arduino.h>

#define MAX_LAYERS 10
#define MAX_NEURONS 10
#define BITSIZE 8

class SpikingNeuralNetworkLayer {
 private:
  int inputs;
  int num_neurons;
  //[MAX_LAYERS][MAX_NEURONS];
  //[MAX_LAYERS][MAX_NEURONS];
  int8_t** weights;  //[MAX_LAYERS-1][MAX_NEURONS][MAX_NEURONS];
  int* spike_counter;

  int threshold = 7;
  int spike_time;
  int time_step;
  float alpha;
  int refractory_period;
  float beta;
  float min_v;
  float max_v;
  float qa;
  float qb;
  float qs;
  float qz;
  int* outputs;

 public:
  float* membrane_potential;
  int8_t* outputspikes;
  int8_t* neuron_state;
  void initialize_layer(int inputs, int num_neurons, float threshold,
                        float alpha, int refractory_period) {
    // this->num_layers = num_layers;
    this->qa = pow(-2, BITSIZE - 1);
    this->qb = pow(2, BITSIZE - 1) - 1;
    this->inputs = inputs;
    this->outputs = num_neurons;
    this->num_neurons = num_neurons;
    this->threshold = threshold;
    this->spike_counter = new int[num_neurons];
    this->alpha = alpha;  // time decay
    this->refractory_period =
        refractory_period;  // how long in the time to reset the neuron
    this->weights = new int8_t*[num_neurons];
    this->neuron_state = new int8_t[num_neurons];
    this->membrane_potential = new float[num_neurons];
    this->outputspikes = new int8_t[num_neurons];
    for (int i = 0; i < num_neurons; i++) {
      this->weights[i] = new int8_t[inputs];
      this->neuron_state[i] = 0;
      this->membrane_potential[i] = 0.0;
      this->outputspikes[i] = 0;
      for (int o = 0; o < inputs; o++) {
        this->weights[i][o] = (int8_t)random(-30, 30);
        // Serial.println(this->weights[i][o]);
      }
    }
  };

  void q_learning(float reward, float learning_rate, float discount_factor,
                  int* prev_state, int* action, int* next_state) {
    // Compute the current Q-value estimate for the previous state and action
    float prev_q_value = 0;
    for (int i = 0; i < num_neurons; i++) {
      prev_q_value += weights[prev_state[i]][action[i]];
    }

    // Compute the maximum Q-value estimate for the next state
    float max_next_q_value = 0;
    for (int i = 0; i < num_neurons; i++) {
      float next_q_value = 0;
      for (int j = 0; j < num_neurons; j++) {
        next_q_value += weights[next_state[i]][j];
      }
      if (next_q_value > max_next_q_value) {
        max_next_q_value = next_q_value;
      }
    }

    // Update the weights using the Q-learning rule
    float td_error = reward + discount_factor * max_next_q_value - prev_q_value;
    for (int i = 0; i < num_neurons; i++) {
      for (int j = 0; j < inputs; j++) {
        weights[prev_state[i]][j] += learning_rate * td_error;
      }
    }
  };

  float dequantize_value(int8_t value) {
    return this->qs * (value - this->qz);
  };
  int8_t quantize_value(float value) {
    int8_t qvalue = round(1 / this->qs * value + this->qz);
    // x_q = np.round(1 / s * arr + z, decimals=0)
    // x_q = np.clip(x_q, a_min=alpha_q, a_max=beta_q)
    return qvalue;
  };
  void generate_quantize_vars(float maxv, float minv) {
    this->min_v = minv;
    this->max_v = maxv;
    this->qs = (maxv - minv) / (this->qb - this->qa);
    this->qz = int((maxv * this->qa - minv * this->qb) / (maxv - minv));
    Serial.print("Quantize variables:\nA:");
    Serial.print(this->qa);
    Serial.print(" b:");
    Serial.print(this->qb);
    Serial.print(" s:");
    Serial.print(this->qs);
    Serial.print(" z:");
    Serial.println(this->qz);
  };
  int get_output() {
    int max_v = 0;
    int max_ind = 0;
    for (int n = 0; n < this->num_neurons; n++) {
      if (this->spike_counter[n] > max_v) {
        max_v = this->spike_counter[n];
        max_ind = n;
      }
    }
    reset();
    return max_ind;
  };
  void reset() {
    for (int n = 0; n < this->num_neurons; n++) {
      if (this->outputspikes[n] >= 1) {
        this->outputspikes[n] = 0;
      }
      this->neuron_state[n] = (int8_t)0;
      this->membrane_potential[n] = 0;
    }
  };
  void print_spikes() {
    for (int n = 0; n < this->num_neurons; n++) {
      Serial.print(this->outputspikes[n]);
      Serial.print(",");
    }
  };
  void print_voltajes() {
    for (int n = 0; n < this->num_neurons; n++) {
      Serial.print(this->membrane_potential[n], 8);
      Serial.print(",");
    }
  };
  
  void forward(byte* inputs) {
    // first read the input spikes
    // iterate over the neurons

    for (int n = 0; n < this->num_neurons; n++) {
      // float tmp_float = 0;
      //  iterate over the inputs
      if (this->outputspikes[n] >= 1) {
        this->outputspikes[n] = 0;
      }
      if (neuron_state[n] == 0) {
        for (int xn = 0; xn < this->inputs; xn++) {
          // if recived an spike, perform the operation
          if (inputs[xn] != 0) {
            // Serial.println("SSWOW");
            this->membrane_potential[n] =
                (this->membrane_potential[n] +
                 (dequantize_value(this->weights[n][xn]))) -
                (0.0005 * this->membrane_potential[n] + 0.001);
          }
          if (this->membrane_potential[n] > this->max_v) {
            this->membrane_potential[n] = this->max_v;
          }
          if (this->membrane_potential[n] < this->min_v) {
            this->membrane_potential[n] = this->min_v;
          }
          if (this->membrane_potential[n] > this->threshold) {
            this->spike_counter[n] = this->spike_counter[n] + 1;
            this->neuron_state[n] = (int8_t)1;
            this->outputspikes[n] = 1;
            this->membrane_potential[n] = -this->threshold;
            //
          }
          // Serial.println(membrane_potential[n]);
        }
        //
      } else {
        if (this->neuron_state[n] > this->refractory_period) {
          this->neuron_state[n] = (int8_t)0;
        } else {
          this->neuron_state[n] = this->neuron_state[n] + (int8_t)1;
        }
      }
    }
  };

void learn(byte* inputs, byte* expected_output) {
// Forward the input and store the state
forward(inputs);
int predicted_output = get_output();
// If the predicted output is different from the expected output
if(predicted_output != expected_output) {
    // Iterate over the neurons in the output layer
    for(int n = 0; n < this->num_neurons; n++) {
        // If the neuron was active in this timestep
        if(this->outputspikes[n] >= 1) {
            // Iterate over the inputs
            for(int xn = 0; xn < this->inputs; xn++) {
                // If the input was active in this timestep
                if(inputs[xn] != 0) {
                    // Perform the weight update
                    int8_t old_weight = this->weights[n][xn];
                    int8_t delta_weight = (int8_t) round(alpha * dequantize_value(this->membrane_potential[n]) * inputs[xn]);
                    this->weights[n][xn] = (int8_t) constrain(old_weight + delta_weight, -30, 30);
                }
            }
        }
    }
}

// Reset the layer state
reset();
};
};

#endif
