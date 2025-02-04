pub mod act_layers;

trait SingleDimLayer<const IN: usize, const OUT: usize> {
    fn new() -> Self;
    fn evaluate(&self, inp: [f64; IN]) -> [f64; OUT];
    fn internal_gradient(&self, inp: [f64; IN], out_grad: &[f64; OUT]) -> Gradient<IN, OUT>;
    fn backprop(&self, inp: [f64; IN], out_grad: &[f64; OUT]) -> [f64; IN];
    fn apply_gradient(&mut self, grad: Gradient<IN, OUT>) -> Result<(), String>;
}
trait SingleDimLayerExtra {}
impl<T> SingleDimLayerExtra for T where T: SingleDimLayer {}

enum Gradient<const X: usize, const Y: usize> {
    SimpleActivation([f64; X]),
    LayerConnection([[f64; Y]; X]),
    Nil,
}

struct SimpleActivationLayer<const X: usize> {
    bias: [f64; X],
}

pub struct LayerConnection<const IN: usize, const OUT: usize> {
    weights: [[f64; OUT]; IN],
}
impl<const IN: usize, const OUT: usize> LayerConnection<IN, OUT> for Net<IN, OUT> {
    fn new() -> Self {
        Net {
            weights: [[0.0; OUT]; IN],
        }
    }

    fn evaluate(&self, inp: [f64; IN]) -> [f64; OUT] {
        let mut out: [f64; OUT] = [0.0];

        self.iter().map(|out_weights| {
            for i in 0..OUT {
                out[i] += inp[i] * out_weights[i]
            }
        });

        out
    }

    fn internal_gradient(&self, inp: [f64; IN], out_gradient: &[f64; OUT]) -> Gradient<IN, OUT> {
        out_gradient


        Gradient::LayerConnection(())
    }

    fn backprop(&self, inp: [f64; IN], out_gradient: &[f64; OUT]) -> [f64; IN] {
        todo!()
    }

    fn apply_gradient(&mut self, grad: Gradient<IN, OUT>) -> Result<(), String> {
        todo!()
    }
}
