pub mod matrix;

use self::matrix::{Vector, PublicAttribute, Vectorizable};

pub fn sigmoid(x : f64) -> f64 {
	1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_prime(x : f64) -> f64 {
	let result_sigmoid = sigmoid(x);

	result_sigmoid * (1.0 - result_sigmoid)
}

pub fn softmax(vector : & Vector) -> Vector {
	let mut sum_of_all_exponentials = 0.0;

	for elem in vector.value() {
		sum_of_all_exponentials += elem.exp();
	}

	let f = |z : f64| z.exp() / sum_of_all_exponentials;
	vector.vectorize(f)
}

pub fn cross_entropy(required_outputs : & Vec<Vec<f64>>, predicted_outputs : & Vec<Vec<f64>>) -> f64 {
	let mut sum = 0.0;

	required_outputs.iter().zip(predicted_outputs).for_each(|(required_output, predicted_output)| {
		for (y, yb) in required_output.iter().zip(predicted_output) {
			sum += (y * yb.ln()) + ((1.0 - y) * (1.0 - yb).ln());
		}
	});

	- sum / required_outputs.len() as f64
}
