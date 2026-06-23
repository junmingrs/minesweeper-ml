use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, prelude::Backend, tensor::activation};

#[derive(Module, Debug)]
pub struct Policy<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Policy<B> {
    pub fn new(device: &B::Device, input: usize, output: usize) -> Self {
        Self {
            fc1: LinearConfig::new(input, 128).init(device),
            fc2: LinearConfig::new(128, output).init(device),
        }
    }
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}
