use super::configuration;
use super::Perceptron;
use crate::maths::matrix::Vector;
use rand::thread_rng;
use rand::Rng;

impl Perceptron {
	pub fn train(& mut self, training_inputs : Vec<Vec<f64>>, training_outputs : Vec<Vec<f64>>) {

		let dataset = Perceptron::get_dataset(training_inputs, training_outputs);

		self.stochastic_gradient_descent(& dataset);
	}

	fn get_dataset(training_inputs : Vec<Vec<f64>>, training_outputs : Vec<Vec<f64>>) -> Vec<(Vector, Vector)> {

		let mut dataset : Vec<(Vector, Vector)> = training_inputs.into_iter().zip(training_outputs.into_iter()).map(|(input, output)| {
			(Vector::new(input), Vector::new(output))
		}).collect();

		let mut rng = thread_rng();
		rng.shuffle(& mut dataset);

		dataset

	}
}