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

	pub fn evaluate_loss(predicted_outputs : Vec<Vec<f64>>, required_outputs : & Vec<Vec<f64>>) -> f64 {

		let mut outputs_comparison : Vec<(Vec<f64>, & Vec<f64>)> = Vec::with_capacity(predicted_outputs.len());

		predicted_outputs.into_iter().zip(required_outputs).for_each(|(predicted_output, required_output)| {
			outputs_comparison.push((predicted_output, & required_output))
		});

		configuration::LOSS(& outputs_comparison)
	}
}