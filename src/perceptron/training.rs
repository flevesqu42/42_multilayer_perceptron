use super::Perceptron;
use crate::maths::matrix::Vector;

use rand::thread_rng;
use rand::Rng;

impl Perceptron {
	pub fn train(& mut self	, inputs : Vec<Vec<f64>>, outputs : Vec<Vec<f64>>)
	{
		let mut training_dataset = Perceptron::get_dataset(inputs, outputs);
		let mut rng = thread_rng();

		rng.shuffle(& mut training_dataset);
		let validation_dataset = training_dataset.split_off(training_dataset.len() / 2);

		self.stochastic_gradient_descent(& mut training_dataset, & validation_dataset);
	}

	fn get_dataset(training_inputs : Vec<Vec<f64>>, training_outputs : Vec<Vec<f64>>) -> Vec<(Vector, Vector)> {

		let dataset : Vec<(Vector, Vector)> = training_inputs.into_iter().zip(training_outputs.into_iter()).map(|(input, output)| {
			(Vector::new(input), Vector::new(output))
		}).collect();

		dataset
	}
}