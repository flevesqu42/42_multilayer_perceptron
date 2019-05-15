mod informations;
mod data;

use std::io::Error;
use self::informations::Informations;
use self::data::Data;

/* Structure */

#[derive(Debug)]
pub struct Dataset {
	pub datas:			Vec<Data>,
	pub describe:		[Informations; 30],
	pub outputs_vector:	Vec<Vec<f64>>
}

/* Implementations */

impl Dataset {

	/* public methods */

	pub fn new(path : & str) -> Result<Dataset, Error> {

		let datas = Dataset::get_datas(path)?;
		let describe = Dataset::get_describe(&datas);
		let outputs_vector = datas.iter().map(|data| data.output.to_vec()).collect();

		Ok(Dataset{datas, describe, outputs_vector})
	}

	pub fn standardize(self) -> Dataset {
		let mut datas : Vec<Data> = vec![];

		for data in self.datas {
			datas.push(data.standardize(& self.describe));
		}

		let describe = Dataset::get_describe(& datas);

		Dataset {datas, describe, outputs_vector: self.outputs_vector}
	}

	pub fn features_count(& self) -> usize {
		self.describe.len()
	}

	pub fn output_size(& self) -> usize {
		self.datas[0].output.len()
	}

	pub fn positives_cases(& self) -> usize {
		let mut count = 0;

		for data in & self.datas {
			count += match data.output {
				[x, _] if x == 1.0	=> 1,
				_					=> 0
			}
		}

		count
	}

	/* private methods */

	fn get_datas(path : & str) -> Result<Vec<Data>, Error> {
		let reader = csv::Reader::from_path(path)?;
		let mut datas : Vec<Data> = vec![];

		for result in reader.into_records() {
			let record = result?;
			let data = Data::new(& record)?;

			datas.push(data);
		}

		Ok(datas)
	}

	fn get_describe(dataset : & [Data]) -> [Informations; 30] {
		let mut informations : [Informations; 30] = Default::default();

		for (idx, information) in informations.iter_mut().enumerate() {
			*information = Informations::new(dataset, idx);
		}

		informations
	}

}