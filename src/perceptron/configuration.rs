use crate::maths::{sigmoid, sigmoid_prime, softmax, matrix::Vector, cross_entropy};

pub static VERBOSE : bool = true;

pub static WIDTH :	usize	= 2;
pub static HEIGHT :	usize	= 4;

pub static LEARNING_RATE : f64 = 0.2;
pub static MINI_BATCH_SIZE : usize = 32;
pub static EPOCHS : usize = 100;

pub static WEIGHT_MIN_RANGE : f64 = -1.0;
pub static WEIGHT_MAX_RANGE : f64 = 1.0;

pub static BIAS_MIN_RANGE : f64 = -1.0;
pub static BIAS_MAX_RANGE : f64 = 1.0;

pub static ACTIVATION		: fn(f64) -> f64 = sigmoid;
pub static ACTIVATION_PRIME : fn(f64) -> f64 = sigmoid_prime;

pub static OUTPUT : fn(& Vector) -> Vector = softmax;

pub static LOSS : fn(& Vec<(Vec<f64>, & Vec<f64>)>) -> f64 = cross_entropy;
