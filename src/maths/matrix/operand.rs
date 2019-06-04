use super::*;

/* Right hand side must always be referenced to keep integrity */

/* PublicAttribute operands */

pub trait MatrixOperand {
	fn hadamard(& self, rhs: & dyn PublicAttribute) -> Self; // keep integrity
	fn hadamard_inplace(self, rhs: & dyn PublicAttribute) -> Self; // without keeping integrity
}

pub trait VectorOperand {
	fn dot(& self, rhs: & dyn PublicAttribute) -> f64;
}

impl MatrixOperand for Matrix {

	fn hadamard(& self, rhs: & dyn PublicAttribute) -> Self {
		assert!(self.col == rhs.col() && self.row == rhs.row());

		let col = self.col;
		let row = self.row;
		let value : Vec<f64> = self.value.iter().zip(rhs.value()).map(|(e1, e2)| {
			*e1 * *e2
		}).collect();

		Self {value, col, row}
	}

	fn hadamard_inplace(mut self, rhs: & dyn PublicAttribute) -> Self {
		assert!(self.col() == rhs.col() && self.row() == rhs.row());

		self.value.iter_mut().zip(rhs.value()).for_each(|(e1, e2)| {
			*e1 *= e2;
		});

		self
	}

}

impl MatrixOperand for Vector {

	fn hadamard(& self, rhs: & dyn PublicAttribute) -> Self {
		assert!(rhs.col() == 1 && self.value.len() == rhs.row());

		let value : Vec<f64> = self.value.iter().zip(rhs.value()).map(|(e1, e2)| {
			*e1 * *e2
		}).collect();

		Self {value}
	}

	fn hadamard_inplace(mut self, rhs: & dyn PublicAttribute) -> Self {
		assert!(rhs.col() == 1 && self.value.len() == rhs.row());

		self.value.iter_mut().zip(rhs.value()).for_each(|(e1, e2)| {
			*e1 *= *e2;
		});

		self
	}
}

/* scalars for vector */

impl ops::Mul<f64> for & Vector {
	type Output = Vector;

	fn mul(self, rhs : f64) -> Vector {
		let value = self.value.iter().map(|e| {
			e * rhs
		}).collect();

		Vector {value}
	}
}

impl ops::Mul<f64> for Vector {
	type Output = Vector;

	fn mul(self, rhs : f64) -> Vector {
		let value = self.value.into_iter().map(|e| {
			e * rhs
		}).collect();

		Vector {value}
	}
}

impl ops::Mul<& Vector> for f64 {
	type Output = Vector;

	fn mul(self, rhs : & Vector) -> Vector {
		rhs * self
	}
}

/* scalars for Matrix */

impl ops::Mul<f64> for & Matrix {
	type Output = Matrix;

	fn mul(self, rhs : f64) -> Matrix {
		let value = self.value.iter().map(|e| {
			e * rhs
		}).collect();
		let row = self.row;
		let col = self.col;

		Matrix {value, row, col}
	}
}

impl ops::Mul<f64> for Matrix {
	type Output = Matrix;

	fn mul(self, rhs : f64) -> Matrix {
		let value = self.value.into_iter().map(|e| {
			e * rhs
		}).collect();
		let row = self.row;
		let col = self.col;

		Matrix {value, row, col}
	}
}

impl ops::Mul<& Matrix> for f64 {
	type Output = Matrix;

	fn mul(self, rhs : & Matrix) -> Matrix {
		rhs * self
	}
}

/* Operator overload for vector */

impl ops::Add<& Vector> for & Vector {
	type Output = Vector;

	fn add(self, rhs : & Vector) -> Vector {
		assert!(self.value.len() == rhs.value.len());

		let value = self.value.iter().zip(& rhs.value).map(|(e1, e2)| {
			*e1 + *e2
		}).collect();

		Vector {value}
	}
}

impl ops::Add<& Vector> for Vector {
	type Output = Vector;

	fn add(mut self, rhs : & Vector) -> Vector {
		assert!(self.value.len() == rhs.value.len());

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 += *e2;
		});

		self
	}
}

impl ops::AddAssign<& Vector> for Vector {

	fn add_assign(& mut self, rhs : & Vector) {
		assert!(self.value.len() == rhs.value.len());

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 += *e2;
		});

	}
}

impl ops::Sub<& Vector> for & Vector {
	type Output = Vector;

	fn sub(self, rhs : & Vector) -> Vector {
		assert!(self.value.len() == rhs.value.len());

		let value : Vec<f64> = self.value.iter().zip(& rhs.value).map(|(e1, e2)| {
			*e1 - *e2
		}).collect();

		Vector {value}
	}
}

impl ops::Sub<& Vector> for Vector {
	type Output = Vector;

	fn sub(mut self, rhs : & Vector) -> Vector {
		assert!(self.value.len() == rhs.value.len());

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 -= *e2;
		});

		self
	}
}

impl ops::SubAssign<& Vector> for Vector {

	fn sub_assign(& mut self, rhs : & Vector) {
		assert!(self.value.len() == rhs.value.len());

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 -= *e2;
		});

	}
}

/* Operator overload for matrix */

impl ops::Add<& Matrix> for & Matrix {
	type Output = Matrix;

	fn add(self, rhs : & Matrix) -> Matrix {
		assert!(self.col == rhs.col && self.row == rhs.row);

		let col = self.col;
		let row = self.row;

		let value : Vec<f64> = self.value.iter().zip(& rhs.value).map(|(e1, e2)| {
			*e1 + *e2
		}).collect();

		Matrix {col, row, value}
	}
}

impl ops::Add<& Matrix> for Matrix {
	type Output = Matrix;

	fn add(mut self, rhs : & Matrix) -> Matrix {
		assert!(self.col == rhs.col && self.row == rhs.row);

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 += *e2;
		});

		self
	}
}

impl ops::AddAssign<& Matrix> for Matrix {

	fn add_assign(& mut self, rhs : & Matrix) {
		assert!(self.col == rhs.col && self.row == rhs.row);

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 += *e2;
		});

	}
}

impl ops::Sub<& Matrix> for & Matrix {
	type Output = Matrix;

	fn sub(self, rhs : & Matrix) -> Matrix {
		assert!(self.col == rhs.col && self.row == rhs.row);

		let col = self.col;
		let row = self.row;

		let value : Vec<f64> = self.value.iter().zip(& rhs.value).map(|(e1, e2)| {
			*e1 - e2
		}).collect();

		Matrix {col, row, value}
	}
}

impl ops::Sub<& Matrix> for Matrix {
	type Output = Matrix;

	fn sub(mut self, rhs : & Matrix) -> Matrix {
		assert!(self.col == rhs.col && self.row == rhs.row);

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 -= e2;
		});

		self
	}
}

impl ops::SubAssign<& Matrix> for Matrix {
	fn sub_assign(& mut self, rhs : & Matrix) {
		assert!(self.col == rhs.col && self.row == rhs.row);

		self.value.iter_mut().zip(& rhs.value).for_each(|(e1, e2)| {
			*e1 -= e2;
		});
	}
}

impl ops::Mul<& Matrix> for & Matrix {
	type Output = Matrix;

	fn mul(self, rhs : & Matrix) -> Matrix {
		assert!(self.col == rhs.row);

		let row = self.row;
		let col = rhs.col;
		let value = (0..row * col).map(|idx| {
			let x = idx % col;
			let y = idx / col;
			let mut val = 0.0;

			(0..self.col).for_each(|k| {
				val += self.value[y * self.col + k] * rhs.value[k * rhs.col + x];
			});

			val
		}).collect();

		Matrix {value, row, col}
	}
}


impl ops::Mul<& Matrix> for Matrix {
	type Output = Matrix;

	fn mul(self, rhs : & Matrix) -> Matrix {
		assert!(self.col == rhs.row);

		let row = self.row;
		let col = rhs.col;
		let value = (0..row * col).map(|idx| {
			let x = idx % col;
			let y = idx / col;
			let mut val = 0.0;

			(0..self.col).for_each(|k| {
				val += self.value[y * self.col + k] * rhs.value[k * rhs.col + x];
			});

			val
		}).collect();

		Matrix {value, row, col}
	}
}

/* Matrix multiplication by vector (m * v) */

impl ops::Mul<& Vector> for & Matrix {
	type Output = Vector;

	fn mul(self, rhs : & Vector) -> Vector {
		assert!(self.col == rhs.value.len());

		let len = self.row;
		let value = (0..len).map(|y| {
			let mut val = 0.0;

			(0..self.col).for_each(|k| {
				val += self.value[y * self.col + k] * rhs.value[k];
			});

			val
		}).collect();

		Vector {value}
	}
}

impl ops::Mul<& Vector> for Matrix {
	type Output = Vector;

	fn mul(self, rhs : & Vector) -> Vector {
		assert!(self.col == rhs.value.len());

		let len = self.row;
		let value = (0..len).map(|y| {
			let mut val = 0.0;

			(0..self.col).for_each(|k| {
				val += self.value[y * self.col + k] * rhs.value[k];
			});

			val
		}).collect();

		Vector {value}
	}
}

/* vector multiplication by matrix (v * m) */

impl ops::Mul<& Matrix> for & Vector {
	type Output = Matrix;

	fn mul(self, rhs : & Matrix) -> Matrix {
		assert!(rhs.row == 1);

		let col = rhs.col;
		let row = self.value.len();
		let value = (0..col * row).map(|idx| {
			let y = idx / col;
			let x = idx % col;

			self.value[y] * rhs.value[x]
		}).collect();

		Matrix {col, row, value}
	}
}

impl ops::Mul<& Matrix> for Vector {
	type Output = Matrix;

	fn mul(self, rhs : & Matrix) -> Matrix {
		assert!(rhs.row == 1);

		let col = rhs.col;
		let row = self.value.len();
		let value = (0..col * row).map(|idx| {
			let y = idx / col;
			let x = idx % col;

			self.value[y] * rhs.value[x]
		}).collect();

		Matrix {col, row, value}
	}
}
