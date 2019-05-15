use crate::maths::{sigmoid, sigmoid_prime, softmax, matrix::Vector};

pub static WIDTH :	usize	= 2;
pub static HEIGHT :	usize	= 5;

pub static WEIGHT_MIN_RANGE : f64 = -3.0;
pub static WEIGHT_MAX_RANGE : f64 = 3.0;

pub static BIAS_MIN_RANGE : f64 = -3.0;
pub static BIAS_MAX_RANGE : f64 = 3.0;

pub static ACTIVATION		: fn(f64) -> f64 = sigmoid;
pub static ACTIVATION_PRIME : fn(f64) -> f64 = sigmoid_prime;

pub static OUTPUT : fn(& Vector) -> Vector = softmax;