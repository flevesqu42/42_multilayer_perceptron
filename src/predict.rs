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

fn make_prediction(mut perceptron : Perceptron, dataset : Dataset) {
	let inputs_vector = dataset.inputs_vector();
	let outputs_vector = dataset.outputs_vector();

	let predictions = perceptron.predict_all(inputs_vector);
	let loss = Perceptron::evaluate_loss(& outputs_vector, & predictions);

	println!("Current configuration loss : {}", loss);
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