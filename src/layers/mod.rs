pub mod act_layers;

trait SingleDimLayer<const IN: usize, const OUT: usize> {
    fn new() -> Self;
    fn evaluate(&self, inp: [f64; IN]) -> [f64; OUT];
    fn internal_gradient(&self, inp: [f64; IN], out_gradient: [f64; OUT]) -> Gradient<IN, OUT>;
    fn backprop(&self, inp: [f64; IN], out_gradient: [f64; OUT]) -> [f64; IN];
    fn apply_gradient(&mut self, grad: Gradient<IN, OUT>) -> Result<(), String>;
}

enum Gradient<const X: usize, const Y: usize> {
    SimpleActivation([f64; X]),
    LayerConnection([[f64; Y]; X]),
    Nil,
}

struct SimpleActivationLayer<const X: usize> {
    bias: [f64; X],
}

struct LayerConnection<const IN: usize, const OUT: usize> {
    weights: [[f64; OUT]; IN],
}

impl<const IN: usize, const OUT: usize> SingleDimLayer<IN, OUT> for LayerConnection<IN, OUT> {
    fn new() -> LayerConnection<IN, OUT> {
        LayerConnection {
            //TODO: MAKE THIS RANDOM
            weights: [[0.0; OUT]; IN],
        }
    }
    fn evaluate(&self, inp: [f64; IN]) -> [f64; OUT] {
        //TODO: MAKE THIS WORK
        [inp[0]; OUT]
    }
    fn internal_gradient(&self, inp: [f64; IN], out: [f64; OUT]) -> Gradient<IN, OUT> {
        Gradient::LayerConnection([[0.0; OUT]; IN])
    }
    fn backprop(&self, inp: [f64; IN], out_gradient: [f64; OUT]) -> [f64; IN] {
        [0.0; IN]
    }
    fn apply_gradient(&mut self, grad: Gradient<IN, OUT>) -> Result<(), String> {
        Ok(())
    }
}
