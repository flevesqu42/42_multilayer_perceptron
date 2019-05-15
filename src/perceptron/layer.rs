use super::configuration;
use crate::maths::matrix::*;

use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Layer {
	pub weights:		Matrix,
	pub bias:			Vector,

	#[serde(skip_serializing, default)]
	pub weighted_input:	Vector,
}

impl Layer {
	pub fn new(input_height : usize, height : usize) -> Layer {
		let mut rng = rand::thread_rng();

		let weights = Layer::gen_weights(input_height, height, & mut rng);
		let bias = Layer::gen_bias(height, & mut rng);
		let weighted_input = Default::default();

		Layer {weights, bias, weighted_input}
	}

	fn gen_weights(input_height : usize, height : usize, rng : & mut rand::prelude::ThreadRng) -> Matrix {
		let values : Vec<Vec<f64>> = (0..height).map(|_| (0..input_height).map(|_|
			rng.gen_range(configuration::WEIGHT_MIN_RANGE, configuration::WEIGHT_MAX_RANGE)
		).collect()).collect();

		Matrix::new(values)
	}

	fn gen_bias(height : usize, rng : & mut rand::prelude::ThreadRng) -> Vector {
		let values : Vec<f64> = (0..height).map(|_|
			rng.gen_range(configuration::BIAS_MIN_RANGE, configuration::BIAS_MAX_RANGE)
		).collect();

		Vector::new(values)
	}

}
