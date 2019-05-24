use super::Perceptron;
use super::configuration;
use crate::maths::matrix::Vector;

impl Perceptron {

	pub fn stochastic_gradient_descent(& mut self, dataset : & Vec<(Vector, Vector)>) {

		let learning_rate = configuration::LEARNING_RATE / configuration::MINI_BATCH_SIZE as f64;

		for _ in 0..configuration::EPOCHS {
			self.run_one_epoch(dataset, learning_rate);
		}
	}

	fn run_one_epoch(& mut self, dataset : & Vec<(Vector, Vector)>, learning_rate : f64) {
		dataset.chunks(configuration::MINI_BATCH_SIZE).for_each(|batch| {
			for (input, output) in batch {
				self.backpropagation(input, output, learning_rate);
			}
		});
	}
	
}