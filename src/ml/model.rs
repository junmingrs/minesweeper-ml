use bevy::ecs::resource::Resource;
use burn::{
    Tensor,
    backend::{Autodiff, Cuda},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, TensorData, activation::log_softmax},
};

use crate::{
    game::Game,
    ml::{
        env::{Environment, Observation},
        policy::Policy,
        transition::Transition,
    },
};

type Backend = Autodiff<Cuda>;

type MyOptim = OptimizerAdaptor<Adam, Policy<Backend>, Backend>;

#[derive(Resource)]
pub struct Model {
    pub game: Game,
    pub device: Device<Backend>,
    pub policy: Policy<Backend>,
    pub transitions: Vec<Transition>,
    pub optim: MyOptim,
    pub baseline: f32,
}

impl Model {
    pub fn new() -> Self {
        let device = Default::default();

        let width = 20;
        let height = 10;
        let action_size = 2 * width * height;
        let input_size = width * height * 3;

        Model {
            game: Game::new(20, 10, 50),
            policy: Policy::new(&device, input_size, action_size),
            device,
            transitions: Vec::new(),
            optim: AdamConfig::new().init(),
            baseline: 0.0,
        }
    }
    pub fn train_step(&mut self) {
        let obs = self.game.to_observation();
        let obs_tensor = obs_to_tensor(&obs, &self.device);

        let logits = self.policy.forward(obs_tensor);

        let mut logits_vec = logits.clone().into_data().to_vec::<f32>().unwrap();
        let mask = self.game.action_mask();

        for (l, m) in logits_vec.iter_mut().zip(mask.iter()) {
            if *m == 0.0 {
                *l = -1e9;
            }
        }

        let max = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits_vec.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

        let r: f32 = rand::random();
        let mut cumulative = 0.0;

        let action = probs
            .iter()
            .enumerate()
            .find_map(|(i, p)| {
                cumulative += p;
                (r < cumulative).then_some(i)
            })
            .unwrap_or(probs.len() - 1);

        let result = self.game.step(action);

        self.transitions.push(Transition {
            observation: obs,
            action,
            reward: result.reward,
            done: result.done,
            log_prob: probs[action].ln(),
        });

        if result.done {
            self.finish_episode();
            self.game.reset();
        }
    }
    fn compute_returns(&self, gamma: f32) -> Vec<f32> {
        let mut returns = Vec::new();
        let mut g = 0.0;
        for transition in self.transitions.iter().rev() {
            g = transition.reward + gamma * g;
            returns.push(g);
        }
        returns.reverse();
        returns
    }
    fn finish_episode(&mut self) {
        let returns = self.compute_returns(0.99);
        let mut returns = returns;
        normalise(&mut returns);

        let mut loss = Tensor::<Backend, 1>::zeros([1], &self.device);
        let entropy_coeff = 0.01;

        for (t, r) in self.transitions.iter().zip(returns.iter()) {
            let obs = obs_to_tensor(&t.observation, &self.device);

            let logits = self.policy.forward(obs);
            let log_probs = log_softmax(logits, 1);
            let entropy = -(log_probs.clone().exp() * log_probs.clone()).sum();

            let action_tensor =
                Tensor::<Backend, 1>::from_floats([t.action as f32], &self.device).int();
            let selected_log_prob = log_probs.select(1, action_tensor).sum();

            let reward = Tensor::<Backend, 1>::from_floats([*r], &self.device);

            let advantage = reward - self.baseline;

            loss = loss - selected_log_prob * advantage;
            loss = loss - entropy_coeff * entropy;
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.policy);
        self.policy = self.optim.step(1e-3, self.policy.clone(), grads);

        self.transitions.clear();
    }
}
fn normalise(v: &mut [f32]) {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    let std = variance.sqrt().max(1e-8);
    for x in v {
        *x = (*x - mean) / std;
    }
}

fn obs_to_tensor(obs: &Observation, device: &Device<Backend>) -> Tensor<Backend, 2> {
    let mut input: Vec<f32> = Vec::new();
    input.extend(&obs.hidden);
    input.extend(&obs.revealed);
    input.extend(&obs.flagged);
    let input_size = obs.hidden.len() + obs.revealed.len() + obs.flagged.len();
    let data = TensorData::new(input, [1, input_size]);
    Tensor::<Backend, 2>::from_floats(data, device)
}
