use crate::layers::types::{Gradient, WeightUpdates};

pub mod adam;
pub mod gd;

pub trait Optimizer<'a, const COL: usize, const ROW: usize, HP> {
    fn scale(&mut self, grad: Gradient<COL, ROW>) -> WeightUpdates<COL, ROW>;
    fn default(hyperparams: &'a HP) -> Self;
}
