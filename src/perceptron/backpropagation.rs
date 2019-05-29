use super::Perceptron;
use super::Layer;

use crate::maths::matrix::*;
use crate::maths::matrix::operand::MatrixOperand;
use super::configuration::ACTIVATION_PRIME;

impl Perceptron {
	pub fn backpropagation(& mut self, input : & Vector, required_output : & Vector, learning_rate : f64) -> Vector {

		let predicted_output = self.feedforward(input);
		self.set_output_error(& predicted_output, required_output);
		self.backpropagate_error();
		self.update_weights_and_bias(input, learning_rate);

		predicted_output
	}

	fn set_output_error(& mut self, predicted_output : & Vector, required_output : & Vector) {
		/*
		  	Error will be strictly equivalent to derivative `dC / dz`.
			in respect to chain rule it will be `∇aC ⊙ activation_prime(z)`
			with C, z and a respectively the loss function, weighted input and outputs vector.

			In cross entropy loss function we can avoid the learning slowdown side effect of chain rule
			this code is attempted to only work with cross entropy loss function at this point.
		*/

		// let activation_prime = & self.layers.last_mut().unwrap().weighted_sum.vectorize(ACTIVATION_PRIME);

		self.layers.last_mut().unwrap().error = predicted_output - required_output // .hadamard_inplace(activation_prime);
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

	fn update_weights_and_bias(& mut self, input : & Vector, learning_rate : f64) {

			let mut input = input;

			for layer in self.layers.iter_mut() {
				input = layer.update_weights_and_bias(input, learning_rate);
			}
	}
}

impl Layer {
	fn update_weights_and_bias(& mut self, input : & Vector, learning_rate : f64) -> & Vector {
			let input_transpose	= & input.transpose();
			let error			= & self.error;

			let rate_of_change_in_weights	= & (error * input_transpose * learning_rate);
			let rate_of_change_in_bias		= & (error * learning_rate);

			self.updated_weights -= rate_of_change_in_weights;
			self.updated_bias	 -= rate_of_change_in_bias;

			& self.output
	}
}