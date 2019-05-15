use super::Perceptron;

impl Perceptron {

	pub fn predict(& mut self, unvectorized_inputs : Vec<f64>) -> Vec<f64> {
		self.feedforward(unvectorized_inputs).into_vec()
	}

	pub fn predict_all(& mut self, all_inputs : Vec<Vec<f64>>) -> Vec<Vec<f64>> {
		all_inputs.into_iter().map(move |input| {
			self.predict(input)
		}).collect()
	}

	pub fn evaluate_predictions(desired_outpus : & Vec<Vec<f64>>, predictions : & Vec<Vec<f64>>) {
		
	}
}