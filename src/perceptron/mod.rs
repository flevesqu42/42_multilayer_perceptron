pub mod configuration;

mod layer;
mod predict;
mod training;
mod feedforward;
mod backpropagation;

use self::layer::Layer;
use serde::{Serialize, Deserialize};
use std::io::{Error, ErrorKind};

#[derive(Serialize, Deserialize, Debug)]
pub struct Perceptron {
	pub layers: Vec<Layer>
}

pub struct Configuration {
	pub width_hidden_layers :	usize,
	pub height_hidden_layers :	usize,
	pub input_height :			usize,
	pub output_height :			usize,
}

impl Configuration {
	pub fn default(input_height: usize, output_height: usize) -> Configuration {
		Configuration
		{
			input_height,
			output_height,
			width_hidden_layers : configuration::WIDTH,
			height_hidden_layers: configuration::HEIGHT
		}
	}
}

impl Perceptron {
	pub fn new(configuration : Configuration) -> Perceptron {
		let mut layers = vec![];

		let input_layer = Layer::new(configuration.input_height, configuration.height_hidden_layers);
		layers.push(input_layer);

		for _ in 1..configuration.width_hidden_layers {
			let hidden_layer = Layer::new(configuration.height_hidden_layers, configuration.height_hidden_layers);
			layers.push(hidden_layer);
		}

		let output_layer = Layer::new(configuration.height_hidden_layers, configuration.output_height);
		layers.push(output_layer);

		Perceptron {layers}
	}

	pub fn serialize(& self) -> Result<String, Error> {
		match serde_yaml::to_string(self) {
			Ok(serialized)	=> Ok(serialized),
			Err(err)		=> Err(Error::new(ErrorKind::InvalidData, err))
		}
	}

	pub fn deserialize(data : & String) -> Result<Perceptron, Error> {
		match serde_yaml::from_str(data.as_str()) {
			Ok(perceptron)	=> Ok(perceptron),
			Err(err)		=> Err(Error::new(ErrorKind::InvalidData, err))
		}
	}
}