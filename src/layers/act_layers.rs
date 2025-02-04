use super::{Gradient, SingleDimLayer};

pub struct ReLu<const DIM: usize> {}
impl<const DIM: usize> SingleDimLayer<DIM, DIM> for ReLu<DIM> {
    fn new() -> ReLu<DIM> {
        ReLu {}
    }
    fn evaluate(&self, mut inp: [f64; DIM]) -> [f64; DIM] {
        for i in 0..DIM {
            inp[i] = {
                if inp[i] < 0.0 {
                    inp[i]
                } else {
                    0.0
                }
            }
        }
        inp
    }

    fn internal_gradient(&self, _inp: [f64; DIM], _out_grad: &[f64; DIM]) -> Gradient<DIM, DIM> {
        Gradient::Nil
    }

    fn backprop(&self, mut inp: [f64; DIM], out_grad: &[f64; DIM]) -> [f64; DIM] {
        for i in 0..DIM {
            inp[i] = {
                if inp[i] < 0.0 {
                    out_grad[i]
                } else {
                    0.0
                }
            }
        }
        inp
    }

    fn apply_gradient(&mut self, grad: Gradient<DIM, DIM>) -> Result<(), String> {
        match grad {
            Gradient::Nil => Ok(()),
            _ => Err("ReLu does not accept a Gradient".to_string()),
        }
    }
    fn blank_gradient(&mut self) -> Gradient<DIM, DIM> {
        Gradient::Nil
    }
}

pub struct SoftMax<const DIM: usize> {}

impl<const DIM: usize> SingleDimLayer<DIM, DIM> for SoftMax<DIM> {
    fn new() -> Self {
        SoftMax::<DIM> {}
    }

    fn evaluate(&self, inp: [f64; DIM]) -> [f64; DIM] {
        //let inp: Vec<f64> = inp.map(|i| i.exp()).to_vec();
        let sum: f64 = inp.iter().sum();
        inp.map(|i| i.exp() / sum)
    }

    fn backprop(&self, mut inp: [f64; DIM], out_grad: &[f64; DIM]) -> [f64; DIM] {
        inp = inp.map(|i| i.exp());
        let sum: f64 = inp.iter().sum();

        inp.into_iter()
            .map(|i| {
                //TODO: POSSIBLE ERROR: CHECK IF THIS CREATES SOME BORROWING PROBLEMS
                (i * (sum - 2.0) + inp.iter().map(|exp| exp * i).sum::<f64>() - (i * i))
                    / (sum * sum)
            })
            .zip(out_grad.into_iter())
            .map(|(out, grad)| out * grad)
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap()
    }

    fn internal_gradient(&self, _inp: [f64; DIM], _out_grad: &[f64; DIM]) -> Gradient<DIM, DIM> {
        Gradient::Nil
    }

    fn apply_gradient(&mut self, grad: Gradient<DIM, DIM>) -> Result<(), String> {
        match grad {
            Gradient::Nil => Ok(()),
            _ => Err("SoftMax does not accept a Gradient".to_string()),
        }
    }
    fn blank_gradient(&mut self) -> Gradient<DIM, DIM> {
        Gradient::Nil
    }
}
