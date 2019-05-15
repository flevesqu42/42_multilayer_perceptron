use super::Perceptron;
use crate::maths::matrix::Vector;

impl Perceptron {
	pub fn compute_backpropagation(& mut self, inputs : & Vector, output : & Vector) {
		self.feedforward(inputs);

		// compute error
		// backpropagate error
		// gradient descent over weight and bias
	}

	fn compute_error(& mut self) {

	}

	fn backpropagate_error(& mut self) {

	}

	fn gradient_descent(& mut self) {

	}
}