pub mod act_layers;

trait SingleDimLayer<const IN: usize, const OUT: usize> {
    fn new() -> Self;
    fn evaluate(&self, inp: [f64; IN]) -> [f64; OUT];
    fn internal_gradient(&self, inp: [f64; IN], out_grad: &[f64; OUT]) -> Gradient<IN, OUT>;
    fn backprop(&self, inp: [f64; IN], out_grad: &[f64; OUT]) -> [f64; IN];
    fn apply_gradient(&mut self, grad: Gradient<IN, OUT>) -> Result<(), String>;
    fn blank_gradient(&mut self) -> Gradient<IN, OUT>;
}
//trait SingleDimLayerExtra {}
//impl<T> SingleDimLayerExtra for T where T: SingleDimLayer {}

enum Gradient<const X: usize, const Y: usize> {
    SimpleActivation([f64; X]),
    LayerConnection([[f64; X]; Y]),
    Nil,
}

struct SimpleActivationLayer<const X: usize> {
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
        Gradient::LayerConnection(out_gradient.map(|row_grad| inp.map(|inp| inp * row_grad)))
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

    fn apply_gradient(&mut self, grad: Gradient<IN, OUT>) -> Result<(), String> {
        todo!()
    }
    fn blank_gradient(&mut self) -> Gradient<IN, OUT> {
        Gradient::LayerConnection([[0.0; IN]; OUT])
    }
}
