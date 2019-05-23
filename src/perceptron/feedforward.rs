use super::Perceptron;
use super::Layer;
use super::configuration;

use crate::maths::matrix::*;

impl Perceptron {

	pub fn feedforward(& mut self, inputs : & Vector) -> Vector {
		let mut inputs = inputs;

		for layer in self.layers.iter_mut() {
			inputs = layer.feedforward(inputs);
		}

		configuration::OUTPUT(inputs)
	}

}

impl Layer {

	fn feedforward(& mut self, input : & Vector) -> & Vector {
		let w = & self.weights;		
		let b = & self.bias;

		self.weighted_sum = (w * input) + b; // activation((w * i) + b)
		self.output = self.weighted_sum.vectorize(configuration::ACTIVATION);

		& self.output
	}

}