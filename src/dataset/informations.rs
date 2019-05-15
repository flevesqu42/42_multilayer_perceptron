use super::Data;

#[derive(Debug, Default)]
pub struct Informations {
	pub size:		usize,
	pub mean:		f64,
	pub std:		f64, // standard deviation
	pub gdv:		f64, // greater deviation
	pub min:		f64,
	pub max:		f64,
	pub range:		f64,
}

impl Informations {
	pub fn new(datas : & [Data], idx : usize) -> Informations {
		let mut mean		: f64 = 0.0;
		let mut min			: f64 = datas[0].features[idx];
		let mut max			: f64 = datas[0].features[idx];

		for data in datas {
			let current = data.features[idx];
			mean += current;
			if current < min {
				min = current;
			}
			if current > max {
				max = current;
			}
		}

		let size	: usize = datas.len();
		mean /= size as f64;
		let range	: f64 = max - min; 
		let std		: f64 = Informations::get_standard_deviation(datas, idx, mean); 
		let gdv		: f64 = if (min - mean).abs() > (max - mean).abs() {(min - mean).abs()} else {(max - mean).abs()};

		Informations {size, mean, std, gdv, min, max, range}
	}

	fn get_standard_deviation(datas : & [Data], idx : usize, mean : f64) -> f64 {
		let mut	std : f64 = 0.0;

		for data in datas {
			std += (data.features[idx] - mean) * (data.features[idx] - mean);
		}

		std.sqrt() / datas.len() as f64
	}
}
