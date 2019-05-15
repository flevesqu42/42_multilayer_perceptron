extern crate rand;
extern crate serde;

mod analytics;
mod dataset;
mod learning;
mod predict;
mod perceptron;
mod maths;
mod configuration;

use std::io::{Error, ErrorKind};


fn main()
{
	match parse_module(std::env::args().collect()) {
		Ok(_)		=> (),
		Err(error)	=> error_handler(error),
	}
}

fn parse_module(args : Vec<String>) -> Result<(), Error> {
	if args.len() < 2 {
		return Err(Error::new(ErrorKind::InvalidInput, configuration::usage()));
	}
	match &args[1][..] {
		"-a" | "-analytics"	=> analytics::run(&args[2..]),
		"-l" | "-learning"	=> learning::run(&args[2..]),
		"-p" | "-predict"	=> predict::run(&args[2..]),
		cmd					=> Err(Error::new(ErrorKind::Other, format!("`{}`: module not found", cmd)))
	}
}

fn error_handler(error : Error) {
	println!("Error: {}", error);
}
