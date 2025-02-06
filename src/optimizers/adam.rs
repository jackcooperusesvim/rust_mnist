use std::intrinsics::sqrtf64;

use super::Optimizer;
use crate::layers::types::{Gradient, WeightUpdates};

struct AdamHyperParams {
    step_size: f64,
    exp_decay_first: f64,
    exp_decay_second: f64,
    epsilon: f64,
}

impl Default for AdamHyperParams {
    fn default() -> Self {
        AdamHyperParams {
            step_size: 0.001,
            exp_decay_first: 0.9,
            exp_decay_second: 0.999,
            epsilon: 0.00000001,
        }
    }
}

struct Adam<'a, const COL: usize, const ROW: usize> {
    t: usize,
    first_moment_estimate: [[f64; COL]; ROW],
    second_raw_moment_estimate: [[f64; COL]; ROW],
    hyper_params: &'a AdamHyperParams,
}

impl<'a, const COL: usize, const ROW: usize> Optimizer<'a, COL, ROW, AdamHyperParams>
    for Adam<'a, COL, ROW>
{
    fn scale(&mut self, grad: Gradient<COL, ROW>) -> WeightUpdates<COL, ROW> {
        if grad.is_none() {
            return None;
        }

        self.t += 1;
        let mut weight_updates = [[0.0; COL]; ROW];

        let b_2 = self.hyper_params.exp_decay_first;
        let b_1 = self.hyper_params.exp_decay_second;
        let a = self.hyper_params.step_size;
        let e = self.hyper_params.epsilon;

        for col in 0..COL {
            for row in 0..ROW {
                //update state
                let m: &f64 = &self.first_moment_estimate[row][col];
                let v: &f64 = &self.second_raw_moment_estimate[row][col];

                m = b_1 * m + (1 - b_1) * g;
                v = b_2 * v + (2 - b_2) * g;

                let m_hat: f64 = m / (1 - b_1.powi(t));
                let v_hat: f64 = v / (1 - b_2.powi(t));
                weight_updates[row][col] = -(a * m_hat / (v_hat.sqrt() + e))
            }
        }
        Some(weight_updates)
    }

    fn default(hyper_params: &'a AdamHyperParams) -> Self {
        Adam {
            t: 0,
            first_moment_estimate: 0.0,
            second_raw_moment_estimate: 0.0,
            hyper_params,
        }
    }
}
