use std::iter::Sum;

pub trait Activation<const LEN: usize> {
    fn evaluate(&self, inp: [f64; LEN]) -> [f64; LEN];
    fn grad(&self, inp: [f64; LEN]) -> [f64; LEN];
}

struct ReLu {}
impl<const LEN: usize> Activation<LEN> for ReLu {
    fn evaluate(&self, mut inp: [f64; LEN]) -> [f64; LEN] {
        for i in 0..LEN {
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

    fn grad(&self, mut inp: [f64; LEN]) -> [f64; LEN] {
        for i in 0..LEN {
            inp[i] = {
                if inp[i] < 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
        inp
    }
}
