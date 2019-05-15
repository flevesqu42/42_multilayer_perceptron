use super::{Perceptron, configuration};
use crate::maths::matrix::Vector;

impl Perceptron {

	pub fn predict(& mut self, unvectorized_inputs : Vec<f64>) -> Vec<f64> {
		let vectorized_inputs = Vector::new(unvectorized_inputs);

		self.feedforward(& vectorized_inputs).into_vec()
	}

	pub fn predict_all(& mut self, all_inputs : Vec<Vec<f64>>) -> Vec<Vec<f64>> {
		all_inputs.into_iter().map(move |input| {
			self.predict(input)
		}).collect()
	}

	pub fn evaluate_loss(required_outputs : & Vec<Vec<f64>>, predicted_outputs : & Vec<Vec<f64>>) -> f64 {
		configuration::LOSS(required_outputs, predicted_outputs)
	}

}