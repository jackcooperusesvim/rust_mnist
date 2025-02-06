use super::Optimizer;
use crate::layers::types::{Gradient, WeightUpdates};

type GradientDescentHyperParam = GradientDescent;
#[derive(Clone)]
struct GradientDescent {
    learning_rate: f64,
}

impl<'a, const COL: usize, const ROW: usize> Optimizer<'a, COL, ROW, Option<GradientDescent>>
    for GradientDescent
{
    fn scale(&mut self, grad: Gradient<COL, ROW>) -> WeightUpdates<COL, ROW> {
        if grad.is_none() {
            return None;
        }
        Some(
            grad.unwrap()
                .map(|row| row.map(|val| val * self.learning_rate)),
        )
    }
    fn default(_hyperparams: &'a Option<GradientDescent>) -> Self {
        match _hyperparams {
            Some(hp) => hp.clone(),
            None => GradientDescent {
                learning_rate: 0.01,
            },
        }
    }
}
