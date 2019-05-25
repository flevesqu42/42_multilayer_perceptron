use super::Perceptron;
use super::configuration::{LEARNING_RATE, MINI_BATCH_SIZE, EPOCHS, LOSS};
use crate::maths::matrix::{Vector, PublicAttribute};

use rand::thread_rng;
use rand::Rng;

impl Perceptron {

	pub fn stochastic_gradient_descent(& mut self, training_dataset : & mut Vec<(Vector, Vector)>) {

		let learning_rate = LEARNING_RATE / MINI_BATCH_SIZE as f64;

		println!("bla");
		for epoch in 0..EPOCHS {
			let loss = self.run_one_epoch(training_dataset, learning_rate);
			println!("epoch {} loss : {}", epoch, loss);
		}
	}

	fn run_one_epoch(& mut self, training_dataset : & mut Vec<(Vector, Vector)>, learning_rate : f64) -> f64 {
		let mut rng = thread_rng();
		let mut outputs_comparison : Vec<(Vec<f64>, & Vec<f64>)> = Vec::with_capacity(training_dataset.len());

		rng.shuffle(training_dataset);

		training_dataset.chunks(MINI_BATCH_SIZE).for_each(|batch| { // for better understanding, not necessary.
			for (input, required_output) in batch {
				let predicted_output = self.backpropagation(input, required_output, learning_rate);
				outputs_comparison.push((predicted_output.into_vec(), required_output.value()));
			}
		});

		LOSS(& outputs_comparison)
	}

}