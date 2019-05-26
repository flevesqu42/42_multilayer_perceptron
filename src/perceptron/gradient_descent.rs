use super::Perceptron;
use super::configuration::{LEARNING_RATE, MINI_BATCH_SIZE, EPOCHS, LOSS};
use crate::maths::matrix::{Vector, PublicAttribute};

use rand::thread_rng;
use rand::Rng;

impl Perceptron {

	pub fn stochastic_gradient_descent(& mut self, training_dataset : & mut Vec<(Vector, Vector)>, validation_dataset : & Vec<(Vector, Vector)>) {

		let learning_rate = LEARNING_RATE / MINI_BATCH_SIZE as f64;

		for epoch in 0..EPOCHS {
			let loss = self.run_one_epoch(training_dataset, learning_rate);
			let validation_loss = self.get_validation_loss(validation_dataset);
			println!("epoch {} - training_loss: {}, validation_loss: {}", epoch, loss, validation_loss);
		}
	}

	fn run_one_epoch(& mut self, training_dataset : & mut Vec<(Vector, Vector)>, learning_rate : f64) -> f64 {
		let mut rng = thread_rng();

		rng.shuffle(training_dataset);

		let outputs_comparison = training_dataset.iter().map(|(input, required_output)| {
			let predicted_output = self.backpropagation(input, required_output, learning_rate);
			(predicted_output.into_vec(), required_output.value())
		}).collect();

		LOSS(& outputs_comparison)
	}

	fn get_validation_loss(& mut self, dataset : & Vec<(Vector, Vector)>) -> f64 {
		let outputs_comparison = dataset.iter().map(|(input, required_output)| {
			let predicted_output = self.feedforward(input);
			(predicted_output.into_vec(), required_output.value())
		}).collect();

		LOSS(& outputs_comparison)
	}

}