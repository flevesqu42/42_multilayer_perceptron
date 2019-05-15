use crate::perceptron::{Perceptron, Configuration};
use crate::dataset::Dataset;
use crate::configuration::{PERCEPTRON_PATH, usage};

use std::io::{Error, ErrorKind};
use std::fs;

struct Parsing<'a> {
	pub dataset_path : & 'a String
}

impl <'a>Parsing<'a> {
	pub fn new(args : & [String]) -> Result<Parsing, Error> {
		if args.len() != 1 {
			return Err(Error::new(ErrorKind::InvalidInput, usage()));
		}

		let dataset_path = &args[0];

		Ok(Parsing{dataset_path})
	}
}

pub fn run(args: & [String]) -> Result<(), Error> {

	let parsing = Parsing::new(args)?;
	let dataset = Dataset::new(parsing.dataset_path.as_str())?;
	let mut perceptron = Perceptron::new(Configuration::default(30, 2));

	train(& mut perceptron, dataset.standardize());
	export(& perceptron)?;

	Ok(())
}

fn train(perceptron : & mut Perceptron, normalized_dataset : Dataset) {
	// TODO
}

fn export(perceptron : & Perceptron) -> Result<(), Error> {
	let data = perceptron.serialize()?;

	fs::write(PERCEPTRON_PATH, data)?;
	println!("Perceptron configuration saved in `{}`", PERCEPTRON_PATH);

	Ok(())
}