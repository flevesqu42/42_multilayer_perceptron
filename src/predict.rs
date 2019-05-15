use crate::perceptron::Perceptron;
use crate::dataset::Dataset;
use crate::configuration::{PERCEPTRON_PATH, usage};

use std::fs::File;
use std::io::{Error, ErrorKind, BufReader, prelude::*};

// use maths::cross_entropy;

struct Parsing<'a> {
	pub dataset_path: & 'a String
}

impl <'a>Parsing<'a> {
	pub fn new(args: &[String]) -> Result<Parsing, Error> {
		if args.len() != 1 {
			return Err(Error::new(ErrorKind::InvalidInput, usage()));
		}

		let dataset_path = & args[0];

		Ok(Parsing{dataset_path})
	}
}

pub fn run(args: & [String]) -> Result<(), Error> {

	let parsing = Parsing::new(args)?;

	let perceptron = load_perceptron(PERCEPTRON_PATH)?;
	let dataset = Dataset::new(parsing.dataset_path.as_str())?;

	make_prediction(perceptron, dataset.standardize());

	Ok(())
}

fn load_perceptron(path_to_configuration : & str) -> Result<Perceptron, Error> {
	let file = match File::open(path_to_configuration) {
		Ok(file)	=> Ok(file),
		Err(_)		=> Err(Error::new(ErrorKind::NotFound, "Perceptron configuration not found, did you run learning module first ?")),
	}?;

	let mut buf_reader = BufReader::new(file);
	let mut serialized = String::new();

	buf_reader.read_to_string(& mut serialized)?;

	let perceptron = Perceptron::deserialize(& serialized)?;
	println!("Perceptron configuration successfully loaded from `{}`", path_to_configuration);

	Ok(perceptron)
}

fn make_prediction(mut perceptron : Perceptron, dataset : Dataset) {
	// dataset.outputs_vector
}

// fn make_prediction(mut perceptron : Perceptron, dataset : Dataset) {
// 	let predictions = perceptron.predict_all(& dataset);
// 	let loss = cross_entropy(& dataset.outputs_vector, & predictions);

// 	println!("Predictions loss: {}", loss);

// 	let mut good_predictions = 0;

// 	for (prediction_idx, prediction) in predictions.iter().enumerate() {
// 		let mut guessed_output_idx = 0;
// 		let mut guessed_output = 0.0;
// 		for (output_idx, output) in prediction.iter().enumerate() {
// 			if *output > guessed_output {
// 				guessed_output_idx = output_idx;
// 				guessed_output = guessed_output;
// 			}
// 		}
// 		if dataset.datas[prediction_idx].output[guessed_output_idx] == 1.0 {
// 			good_predictions += 1;
// 		}
// 	}

// 	println!("Prediction accuracy: {}% ({} / {})", good_predictions as f32 / dataset.datas.len() as f32, good_predictions, dataset.datas.len());
// }
