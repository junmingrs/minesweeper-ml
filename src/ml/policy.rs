use burn::{
    Tensor,
    module::Module,
    nn::{
        Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::activation,
};

#[derive(Module, Debug)]
pub struct Policy<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    fc1: Linear<B>,
    fc_actor: Linear<B>,
    fc_critic: Linear<B>,
}

impl<B: Backend> Policy<B> {
    pub fn new(device: &B::Device, height: usize, width: usize, action_size: usize) -> Self {
        Self {
            conv1: Conv2dConfig::new([3, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            fc1: LinearConfig::new(64 * height * width, 256).init(device),
            fc_actor: LinearConfig::new(256, action_size).init(device),
            fc_critic: LinearConfig::new(256, 1).init(device),
        }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // let x = activation::relu(self.fc1.forward(x));
        // self.fc2.forward(x)
        let x = activation::relu(self.conv1.forward(x));
        let x = activation::relu(self.conv2.forward(x));
        let x = activation::relu(self.conv3.forward(x));
        // println!("aft conv: {:?}", dims);
        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, c * h * w]);
        let x = activation::relu(self.fc1.forward(x));
        let logits = self.fc_actor.forward(x.clone());
        let value = self.fc_critic.forward(x);
        (logits, value)
    }
}
