use super::Perceptron;
use super::Layer;
use super::configuration;

use crate::maths::matrix::*;

impl Perceptron {

	pub fn feedforward(& mut self, vectorized_inputs : & Vector) -> Vector {
		let mut inputs = vectorized_inputs;

		for layer in self.layers.iter_mut() {
			inputs = layer.feedforward(inputs);
		}

		configuration::OUTPUT(inputs)
	}

}

impl Layer {

	pub fn feedforward(& mut self, input : & Vector) -> & Vector {
		let w = & self.weights;		
		let b = & self.bias;

		self.weighted_input = ((w * input) + b).vectorize_inplace(configuration::ACTIVATION); // activation((w * i) + b)

		& self.weighted_input
	}

}