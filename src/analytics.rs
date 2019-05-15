use std::io::{Error, ErrorKind};
use crate::dataset::Dataset;
use crate::configuration::usage;

pub fn run(args : &[String]) -> Result<(), Error> {

	if args.len() != 1 {
		return Err(Error::new(ErrorKind::InvalidInput, usage()));
	}

	let dataset = Dataset::new(&args[0])?;
	display_describe(& dataset);

	Ok(())
}

fn display_describe(dataset : & Dataset) {
	println!("{:10} {:10} {:35} {:22} {:22} {:20} {:20} {:20}", "feature", "count", "mean", "std", "gdv", "min", "max", "range");
	for i in 0..dataset.describe.len() {
		println!("{:<10} {:<10} {:<35} {:<22} {:<22} {:<20} {:<20} {}", i, dataset.describe[i].size, dataset.describe[i].mean, dataset.describe[i].std, dataset.describe[i].gdv, dataset.describe[i].min, dataset.describe[i].max, dataset.describe[i].range);
	}
	println!("\nThere are {} positives cases in this dataset.", dataset.positives_cases());
}
