use std::io::{Error, ErrorKind};
use super::Informations;

#[derive(Debug)]
pub struct Data {
	pub id:			String,
	pub output:		[f64; 2],	// [Malignant, Begnin]
	pub features:	[f64; 30]
}

impl Data {
	pub fn new(datas : & csv::StringRecord) -> Result<Data, Error> {
		let id = datas[0].to_string();
		let output = match &datas[1] {
			"M"	=> [1.0, 0.0],
			_	=> [0.0, 1.0],
			// _	=> return Err(Error::new(ErrorKind::InvalidData, "unknown diagnosis"))
		};
		let features = Data::get_features(datas)?;

		Ok(Data {id, output, features})
	}

	fn get_features(datas : & csv::StringRecord) -> Result<[f64; 30], Error> {
		let mut features : [f64; 30] = Default::default();

		if datas.len() != 32 {
			return Err(Error::new(ErrorKind::InvalidInput, "bad csv input"));
		}

		for i in 2..32 {
			features[i - 2] = datas[i].parse::<f64>().unwrap();
		}

		Ok(features)
	}

	pub fn standardize(self, describe : & [Informations; 30]) -> Data {
		let mut features : [f64; 30] = Default::default();

		for idx in 0..30 {
			features[idx] = (self.features[idx] - describe[idx].mean) / describe[idx].std;
		}

		Data {id: self.id, output: self.output, features}
	}
}