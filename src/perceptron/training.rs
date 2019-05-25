use super::Perceptron;
use crate::maths::matrix::Vector;

impl Perceptron {
	pub fn train(& mut self, training_inputs : Vec<Vec<f64>>, training_outputs : Vec<Vec<f64>>) {

		let mut dataset = Perceptron::get_dataset(training_inputs, training_outputs);

		self.stochastic_gradient_descent(& mut dataset);
	}

	fn get_dataset(training_inputs : Vec<Vec<f64>>, training_outputs : Vec<Vec<f64>>) -> Vec<(Vector, Vector)> {

		let dataset : Vec<(Vector, Vector)> = training_inputs.into_iter().zip(training_outputs.into_iter()).map(|(input, output)| {
			(Vector::new(input), Vector::new(output))
		}).collect();

		dataset
	}
}