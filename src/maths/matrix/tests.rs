use super::*;

#[test]
#[should_panic(expected = "assertion failed: rows.len() > 0 && rows[0].len() > 0")]
fn new_matrix_empty_col() {
	let matrix = Matrix::new(vec![vec![]]);
}

#[test]
#[should_panic(expected = "assertion failed: rows.len() > 0 && rows[0].len() > 0")]
fn new_matrix_empty_row() {
	let matrix = Matrix::new(vec![]);
}

#[test]
fn new_matrix() {
	let matrix = Matrix::new(vec![vec![1.0, 2.4, 5.0]]);
	assert!(matrix == Matrix {col: 3, row: 1, value: vec![1.0, 2.4, 5.0]});

	let matrix = Matrix::new(vec![
			vec![1.0, 2.4, 5.0],
			vec![2.0, 3.1, 4.0],
		]);
	assert!(matrix == Matrix {col: 3, row: 2, value: vec![1.0, 2.4, 5.0, 2.0, 3.1, 4.0]});
}

#[test]
#[should_panic(expected = "value.len() > 0")]
fn new_vector_empty() {
	let vector = Vector::new(vec![]);
}

#[test]
fn new_vector() {
	let vector = Vector::new(vec![1.0, 2.4, 5.0]);
	assert!(vector == Vector {value: vec![1.0, 2.4, 5.0]});
}

#[test]
fn vector_transpose() {
	let vector = Vector::new(vec![1.0, 2.4, 5.0]);
	let transpose = Matrix::new(vec![vec![1.0, 2.4, 5.0]]);

	assert!(vector.transpose() == transpose);
}

#[test]
fn matrix_transpose_1r() {
	let matrix = Matrix::new(vec![vec![1.0, 2.4, 5.0]]);
	let transpose = Matrix::new(vec![
		vec![1.0],
		vec![2.4],
		vec![5.0],
	]);

	assert!(matrix.transpose() == transpose);
	assert!(transpose.transpose() == matrix);
}

#[test]
fn matrix_transpose_2r() {
	let matrix = Matrix::new(vec![
		vec![1.0, 2.4, 5.0],
		vec![3.4, 4.2, 0.0],
	]);
	let transpose = Matrix::new(vec![
		vec![1.0, 3.4],
		vec![2.4, 4.2],
		vec![5.0, 0.0],
	]);

	assert!(matrix.transpose() == transpose);
	assert!(transpose.transpose() == matrix);
}

#[test]
fn matrix_transpose_3r() {
	let matrix = Matrix::new(vec![
		vec![1.0, 2.4, 5.0],
		vec![3.4, 4.2, 0.0],
		vec![3.1, 4.3, 6.0],
	]);
	let transpose = Matrix::new(vec![
		vec![1.0, 3.4, 3.1],
		vec![2.4, 4.2, 4.3],
		vec![5.0, 0.0, 6.0],
	]);

	assert!(matrix.transpose() == transpose);
	assert!(transpose.transpose() == matrix);
}

#[test]
fn matrix_transpose_4r() {
	let matrix = Matrix::new(vec![
		vec![1.0, 2.4, 5.0],
		vec![3.4, 4.2, 0.0],
		vec![3.1, 4.3, 6.0],
		vec![5.1, 6.3, 7.0],
	]);
	let transpose = Matrix::new(vec![
		vec![1.0, 3.4, 3.1, 5.1],
		vec![2.4, 4.2, 4.3, 6.3],
		vec![5.0, 0.0, 6.0, 7.0],
	]);

	assert!(matrix.transpose() == transpose);
	assert!(transpose.transpose() == matrix);
}