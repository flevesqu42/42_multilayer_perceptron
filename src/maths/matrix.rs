pub mod operand;

use std::ops;
use serde::{Serialize, Deserialize};

/* STRUCTURES */

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Matrix {
	col:			usize,
	row:			usize,
	value:			Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Vector {
	value:	Vec<f64>,
}

impl Matrix {
	pub fn new(mut rows : Vec<Vec<f64>>) -> Matrix {
		assert!(rows.len() > 0);

		let row = rows.len();
		let col = rows[0].len();
		let mut value : Vec<f64> = vec![];

		for row in rows.iter_mut() {
			assert!(row.len() == col);
			value.append(row);
		}

		Matrix {row, col, value}
	}

	pub fn transposed(& self) -> Matrix {

		let col = self.row;
		let row = self.col;
		let value = (0..self.value.len()).map(|idx| {
			let y = idx % self.row;
			let x = idx / self.row;

			self.value[y * self.col + x]
		}).collect();

		Matrix {value, col, row}
	}

	pub fn into_vec(self) -> Vec<Vec<f64>> {
		(0..self.row).map(|y| {
			let pos = y * self.col;
			self.value[pos..pos + self.col].to_vec()
		}).collect()
	}
}

impl Vector {
	pub fn new(value : Vec<f64>) -> Vector {
		Vector {value}
	}

	pub fn into_vec(self) -> Vec<f64> {
		self.value
	}
}

pub trait PublicAttribute {
	fn col(& self) -> usize;
	fn row(& self) -> usize;
	fn value(& self) -> & Vec<f64>;
}

pub trait Vectorizable {
	fn vectorize(& self, f : impl Fn(f64) -> f64) -> Self; // keep integrity
	fn vectorize_inplace(self, f : impl Fn(f64) -> f64) -> Self; // without keeping integrity
}

impl PublicAttribute for Matrix {
	fn col(& self) -> usize {
		self.col
	}

	fn row(& self) -> usize {
		self.row
	}

	fn value(& self) -> & Vec<f64> {
		& self.value
	}
}

impl PublicAttribute for Vector {
	fn col(& self) -> usize {
		1
	}

	fn row(& self) -> usize {
		self.value.len()
	}

	fn value(& self) -> & Vec<f64> {
		& self.value
	}
}

impl Vectorizable for Vector {
	fn vectorize(& self, f : impl Fn(f64) -> f64) -> Self {

		let value = self.value.iter().map(|e| {
			f(*e)
		}).collect();

		Vector {value}
	}

	fn vectorize_inplace(mut self, f : impl Fn(f64) -> f64) -> Self {

		self.value.iter_mut().for_each(|e| {
			*e = f(*e);
		});

		self
	}
}

impl Vectorizable for Matrix {
	fn vectorize(& self, f : impl Fn(f64) -> f64) -> Self {

		let col = self.col;
		let row = self.row;
		let value = self.value.iter().map(|e| {
			f(*e)
		}).collect();

		Matrix {col, row, value}
	}

	fn vectorize_inplace(mut self, f : impl Fn(f64) -> f64) -> Self {

		self.value.iter_mut().for_each(|e| {
			*e = f(*e);
		});

		self
	}

}
