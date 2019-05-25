pub mod matrix;

use self::matrix::{Vector, PublicAttribute, Vectorizable};

pub fn sigmoid(z : f64) -> f64 {
	1.0 / (1.0 + (-z).exp())
}

pub fn sigmoid_prime(z : f64) -> f64 {
	let result_sigmoid = 1.0 / (1.0 + (-z).exp());

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

pub fn cross_entropy(outputs_comparison : & Vec<(Vec<f64>, & Vec<f64>)>) -> f64 {
	let mut sum = 0.0;

	outputs_comparison.iter().for_each(|(predicted_output, required_output)| {
		for (yb, y) in predicted_output.iter().zip(required_output.iter()) {
			println!("{} {}", y, yb);
			sum += (y * yb.ln()) + ((1.0 - y) * (1.0 - yb).ln());
		}
	});

	- sum / outputs_comparison.len() as f64
}