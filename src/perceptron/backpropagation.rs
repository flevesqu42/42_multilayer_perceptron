use super::Perceptron;
use super::Layer;

use crate::maths::matrix::*;
use crate::maths::matrix::operand::MatrixOperand;
use super::configuration::ACTIVATION_PRIME;

impl Perceptron {
	pub fn compute_backpropagation(& mut self, inputs : & Vector, required_output : & Vector) {

		let predicted_output = self.feedforward(inputs);

		self.set_output_error(& predicted_output, required_output);
		self.backpropagate_error();
	}

	fn set_output_error(& mut self, predicted_output : & Vector, required_output : & Vector) {
		/*
		  	Error will be strictly equivalent to derivative `dC / dz`.
			in respect to chain rule it will be `∇aC ⊙ activation_prime(z)`
			with C, z and a respectively the loss function, weighted inputs and outputs vector.

			In cross entropy loss function we can avoid the learning slowdown side effect of chain rule
			this code is attempted to only work with cross entropy loss function at this point.
		*/

		self.layers.last_mut().unwrap().error = predicted_output - required_output;
	}

	fn backpropagate_error(& mut self) {

		/*
			In meaning, we apply error (dC / dz) backward to the network, scaled at weight and with respect to chain rule.
		*/

		for l in (0..self.layers.len() - 1).rev() {

			let next_layer_weights = & self.layers[l + 1].weights;
			let next_layer_error = & self.layers[l + 1].error;
			let weighted_sum = & self.layers[l].weighted_sum;
			let activation_prime = weighted_sum.vectorize(ACTIVATION_PRIME);

			self.layers[l].error = (next_layer_weights.transpose() * next_layer_error).hadamard_inplace(& activation_prime);
		}
	}

	fn gradient_descent(& mut self, inputs : & Vector, learning_rate : f64) {

			let mut inputs = inputs;

			for layer in self.layers.iter_mut() {
				inputs = layer.update_weights_and_bias(inputs, learning_rate);
			}
	}
}

impl Layer {
	fn update_weights_and_bias(& mut self, inputs : & Vector, learning_rate : f64) -> & Vector {
			let inputs_transpose = & inputs.transpose();
			let weights = & self.weights;
			let bias = & self.bias;
			let error = & self.error;

			let rate_of_change_in_weights	= & ((error * inputs_transpose) * learning_rate);
			let rate_of_change_in_bias		= & (error * learning_rate);

			self.weights	= weights - rate_of_change_in_weights;
			self.bias		= bias - rate_of_change_in_bias;

			& self.output
	}
}