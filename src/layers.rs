pub mod act_layers;
pub mod basic_layers;
pub mod types {
    pub trait SingleDimLayer<const IN: usize, const OUT: usize> {
        fn new() -> Self;
        fn evaluate(&self, inp: [f64; IN]) -> [f64; OUT];
        fn internal_gradient(&self, inp: [f64; IN], out_grad: &[f64; OUT]) -> Gradient<IN, OUT>;
        fn backprop(&self, inp: [f64; IN], out_grad: &[f64; OUT]) -> [f64; IN];
        fn apply_gradient(&mut self, grad: Gradient<IN, OUT>) -> Result<(), String>;
        fn blank_gradient(&mut self) -> Gradient<IN, OUT>;
    }
    pub type Gradient<const COL: usize, const ROW: usize> = Option<[[f64; COL]; ROW]>;

    pub type WeightUpdate<const COL: usize, const ROW: usize> = Option<[[f64; COL]; ROW]>;
}
