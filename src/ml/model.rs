use std::{path::Path, sync::mpsc::Sender};

use bevy::ecs::resource::Resource;
use burn::{
    Tensor,
    backend::{Autodiff, Cuda},
    module::Module,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Device, Int, TensorData, activation::log_softmax},
};

use crate::{
    game::Game,
    ml::{
        env::{Environment, Observation},
        policy::Policy,
        transition::Transition,
    },
    tui::Metric,
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
    pub tx: Sender<Metric>,

    pub episode_count: usize,
    pub last_win: bool,
    pub last_total_reward: f32,
    pub last_loss: f32,
    pub last_steps: usize,
}

impl Model {
    pub fn new(tx: Sender<Metric>) -> Self {
        let device = Default::default();

        let height = 20;
        let width = 10;
        let action_size = width * height;

        Model {
            game: Game::new(20, 10, 50),
            policy: Policy::new(&device, height, width, action_size),
            device,
            transitions: Vec::new(),
            optim: AdamConfig::new().init(),
            tx,
            episode_count: 0,
            last_win: false,
            last_total_reward: 0.0,
            last_loss: 0.0,
            last_steps: 0,
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
            self.episode_count += 1;
            self.finish_episode();
            self.game.reset();

            self.tx.send(Metric::EpisodeDone {
                episode: self.episode_count,
                total_reward: self.last_total_reward,
                steps: self.last_steps,
                win: self.last_win,
                loss: self.last_loss,
            }).unwrap();
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

        let entropy_coeff = 0.01;

        let obs_list: Vec<Tensor<Backend, 4>> = self
            .transitions
            .iter()
            .map(|t| obs_to_tensor(&t.observation, &self.device))
            .collect();

        let obs_batch = Tensor::cat(obs_list, 0);
        // println!("obs_batch shae: {:?}", obs_batch.dims());
        let logits = self.policy.forward(obs_batch);
        let log_probs = log_softmax(logits, 1);

        let actions: Vec<i32> = self.transitions.iter().map(|t| t.action as i32).collect();
        let action_tensor = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(actions, [self.transitions.len()]),
            &self.device,
        );
        // let selected_log_prob = log_probs.select(1, action_tensor).sum();
        let selected_log_probs = log_probs
            .clone()
            .gather(1, action_tensor.unsqueeze_dim(1))
            .squeeze_dim(1);

        // let entropy = -(log_probs.clone().exp() * log_probs.clone()).sum();
        let entropy: Tensor<Backend, 1> = -(log_probs.clone().exp() * log_probs)
            .sum_dim(1)
            .squeeze_dim(1);

        let returns_tensor = Tensor::<Backend, 1>::from_data(
            TensorData::new(returns.clone(), [returns.len()]),
            &self.device,
        );

        let loss =
            (selected_log_probs * returns_tensor).sum().neg() - entropy.sum() * entropy_coeff;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.policy);
        self.policy = self.optim.step(1e-3, self.policy.clone(), grads);

        // for ratatui
        self.last_loss = loss.clone().into_scalar();
        self.last_win = self
            .transitions
            .last()
            .map(|t| t.reward >= 1.0)
            .unwrap_or(false);
        self.last_total_reward = self.transitions.iter().map(|t| t.reward).sum();
        self.last_steps = self.transitions.len();

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

fn obs_to_tensor(obs: &Observation, device: &Device<Backend>) -> Tensor<Backend, 4> {
    let mut input: Vec<f32> = Vec::new();
    input.extend(&obs.hidden);
    input.extend(&obs.revealed);
    input.extend(&obs.hints.iter().map(|h| h / 8.0).collect::<Vec<_>>());
    // println!("obs height={} width={}", obs.height, obs.width);
    let data = TensorData::new(input, [1, 3, obs.height, obs.width]);
    Tensor::<Backend, 4>::from_floats(data, device)
}

pub fn load_model(tx: Sender<Metric>) -> Model {
    println!("loading model");
    let device = Device::<Backend>::default();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let height = 20;
    let width = 10;
    // let action_size = 2 * width * height;
    let action_size = width * height;
    let policy = Policy::new(&device, height, width, action_size)
        .load_file(Path::new("../model.bpk"), &recorder, &device)
        .unwrap();
    Model {
        game: Game::new(height, width, 50),
        policy,
        device,
        transitions: Vec::new(),
        optim: AdamConfig::new().init(),
        tx,
        episode_count: 0,
        last_win: false,
        last_total_reward: 0.0,
        last_loss: 0.0,
        last_steps: 0,
    }
}

pub fn save_model(policy: &Policy<Backend>) {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    policy
        .clone()
        .save_file(Path::new("../model.bpk"), &recorder)
        .unwrap();
    println!("Model saved");
}
