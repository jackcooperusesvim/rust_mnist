use crate::layers::types::*;

pub trait Optimizer<const COL: usize, const ROW: usize> {
    fn scale(&mut self, grad: Gradient<COL, ROW>) -> WeightUpdate<COL, ROW>;
}

struct SGD {}
