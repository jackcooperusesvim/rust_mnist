use crate::optimizers::Optimizer;

use super::types::*;

struct SimpleBias<const X: usize> {
    bias: [f64; X],
}

pub struct LayerConnection<const IN: usize, const OUT: usize> {
    weights: [[f64; IN]; OUT],
}

impl<const IN: usize, const OUT: usize> SingleDimLayer<IN, OUT> for LayerConnection<IN, OUT> {
    fn new() -> Self {
        LayerConnection {
            weights: [[0.0; IN]; OUT],
        }
    }

    fn evaluate(&self, inp: [f64; IN]) -> [f64; OUT] {
        self.weights
            .iter()
            .map(|row| row.iter().zip(inp.iter()).map(|(a, b)| a * b).sum())
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap()
    }

    fn internal_gradient(&self, inp: [f64; IN], out_gradient: &[f64; OUT]) -> Gradient<IN, OUT> {
        Some(out_gradient.map(|row_grad| inp.map(|inp| inp * row_grad)))
    }

    fn backprop(&self, _inp: [f64; IN], out_gradient: &[f64; OUT]) -> [f64; IN] {
        let mut out: [f64; IN] = [0.0; IN];

        for col in 0..IN {
            out[col] = 0.0;
            for row in 0..OUT {
                out[col] += self.weights[col][row]
            }
        }

        // Multiply Gradients
        out.iter()
            .zip(out_gradient.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap()
    }

    fn apply_gradient(&mut self, grad: Gradient<IN, OUT>, opt: &Optimizer) -> Result<(), String> {
        let updates: [[f64; IN]; OUR] = opt.scale(grad).unwrap();
        for col in 0..IN {
            for row in 0..OUT {
                self.weights[IN][OUT] = self.weights[IN][OUT] + updates[IN][OUT];
            }
        }
        return Ok(());
    }
    fn blank_gradient(&mut self) -> Gradient<IN, OUT> {
        Some([[0.0; IN]; OUT])
    }
}
