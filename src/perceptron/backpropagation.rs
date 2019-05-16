use super::Perceptron;
use crate::maths::matrix::*;
use crate::maths::matrix::operand::MatrixOperand;
use super::configuration::ACTIVATION_PRIME;

impl Perceptron {
	pub fn compute_backpropagation(& mut self, inputs : & Vector, required_output : & Vector) {

		let output = self.feedforward(inputs);


		// compute and backpropagate error
		// gradient descent over weight and bias
	}

	fn compute_and_backpropagate_error(& mut self, predicted_output : & Vector, required_output : & Vector) {

		self.layers.last_mut().unwrap().error = predicted_output - required_output; // work only with cross entropy

		for l in (0..self.layers.len() - 1).rev() {
			let wt = self.layers[l + 1].weights.transposed();
			let e = & self.layers[l + 1].error;
			let z = & self.layers[l].weighted_input;
			let dz = & z.vectorize(ACTIVATION_PRIME);

			self.layers[l].error = (wt * e).hadamard_inplace(dz)
		}
	}

	fn gradient_descent(& mut self) {

	}
}