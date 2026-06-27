use std::{path::Path, sync::mpsc::Sender};

use bevy::ecs::resource::Resource;
use burn::{
    Tensor,
    backend::{Autodiff, Cuda},
    module::{AutodiffModule, Module},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Device, Int, TensorData, backend::AutodiffBackend},
};
use rand::RngExt;

use crate::{
    game::Game,
    ml::{
        env::{Environment, Observation},
        policy::Policy,
        replay::ReplayBuffer,
        transition::Transition,
    },
    tui::Metric,
};

type Backend = Autodiff<Cuda>;
type InnerBackend = <Backend as AutodiffBackend>::InnerBackend;

type MyOptim = OptimizerAdaptor<Adam, Policy<Backend>, Backend>;

#[derive(Resource)]
pub struct Model {
    pub games: Vec<Game>,
    pub device: Device<Backend>,
    pub policy: Policy<Backend>,
    pub target: Policy<Backend>,
    pub replay_buffer: ReplayBuffer,
    // pub transitions: Vec<Vec<Transition>>,
    pub optim: MyOptim,
    pub tx: Sender<Metric>,
    pub step_count: usize,

    // hyperparams
    pub target_update_freq: usize,

    pub episode_count: usize,
    pub episode_rewards: Vec<f32>,
    pub last_win: bool,
    pub last_total_reward: f32,
    pub last_loss: f32,
    pub last_steps: usize,
}

impl Model {
    pub fn new(tx: Sender<Metric>) -> Self {
        let num_games = 8;
        let device = Default::default();

        let height = 5; // 20
        let width = 5; // 10
        let action_size = width * height;

        Model {
            games: (0..num_games)
                .map(|_| Game::new(height, width, 5))
                .collect(),
            policy: Policy::new(&device, height, width, action_size),
            target: Policy::new(&device, height, width, action_size),
            replay_buffer: ReplayBuffer::new(50_000),
            device,
            // transitions: (0..num_games).map(|_| Vec::new()).collect(),
            optim: AdamConfig::new().init(),
            tx,
            step_count: 0,
            target_update_freq: 1000,
            episode_count: 0,
            episode_rewards: (0..num_games).map(|_| 0.0).collect(),
            last_win: false,
            last_total_reward: 0.0,
            last_loss: 0.0,
            last_steps: 0,
        }
    }

    fn train_on_batch(&mut self, batch_size: usize) {
        if self.replay_buffer.len() < batch_size {
            return; // wait until buffer has enough
        }

        let height = self.games[0].height;
        let width = self.games[0].width;

        let batch = self.replay_buffer.sample(batch_size);

        // current Q values
        let obs_data: Vec<f32> = batch.iter().flat_map(|t| t.obs.iter().copied()).collect();
        // let obs_batch = Tensor::cat(
        //     batch
        //         .iter()
        //         .map(|t| obs_to_tensor(&t.observation, &self.device))
        //         .collect(),
        //     0,
        // );
        let obs_batch = Tensor::<Backend, 4>::from_floats(
            TensorData::new(obs_data, [batch_size, 3, height, width]),
            &self.device,
        );

        let next_obs_data: Vec<f32> = batch
            .iter()
            .flat_map(|t| t.next_obs.iter().copied())
            .collect();
        let next_obs_batch = Tensor::<Backend, 4>::from_floats(
            TensorData::new(next_obs_data.clone(), [batch_size, 3, height, width]),
            &self.device,
        );

        // let next_obs_batch_inner = Tensor::<InnerBackend, 4>::from_floats(
        //     TensorData::new(next_obs_data, [batch_size, 3, height, width]),
        //     &self.device.clone().inner(),
        // );

        // target Q values using target network
        // let next_obs_batch = Tensor::cat(
        //     batch
        //         .iter()
        //         .map(|t| obs_to_tensor(&t.next_observation, &self.device))
        //         .collect(),
        //     0,
        // );
        let (q_values, _) = self.policy.forward(obs_batch); // [B, action_size]
        let (next_q_online, _) = self.policy.forward(next_obs_batch.clone());

        let next_actions = next_q_online.detach().argmax(1);

        let (next_q_target, _) = self.target.valid().forward(next_obs_batch.inner()); // [B, action_size]
        let next_q_target = Tensor::<Backend, 2>::from_inner(next_q_target);

        // Bellman: Q(s,a) = r + gamma * max(Q(s', a')) * (1 - done)
        let max_next_q = next_q_target.gather(1, next_actions).squeeze_dim(1);
        // let inner_device = next_obs_batch_inner.device();

        let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let dones: Vec<f32> = batch
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();

        let rewards_tensor =
            Tensor::<Backend, 1>::from_data(TensorData::new(rewards, [batch_size]), &self.device);
        let dones_tensor =
            Tensor::<Backend, 1>::from_data(TensorData::new(dones, [batch_size]), &self.device);

        let target_q =
            (rewards_tensor + max_next_q.detach() * (dones_tensor.neg() + 1.0) * 0.99).detach();

        // let target_q = Tensor::<Backend, 1>::from_inner(target_q);

        // select Q values for actions taken
        let actions: Vec<i32> = batch.iter().map(|t| t.action as i32).collect();
        let action_tensor = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(actions, [batch_size]),
            &self.device,
        );
        let selected_q = q_values
            .gather(1, action_tensor.unsqueeze_dim(1))
            .squeeze_dim(1); // [B]

        // MSE loss between predicted and target Q values
        let loss: Tensor<Backend, 1> = (selected_q - target_q.detach()).powf_scalar(2.0).mean();
        self.last_loss = loss.clone().into_scalar();

        // NOTE: might need a is_nan / is_infinite check here for last_loss

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.policy);
        self.policy = self.optim.step(1e-4, self.policy.clone(), grads);

        // periodically sync target network
        self.step_count += 1;
        if self.step_count % self.target_update_freq == 0 {
            self.target = self.policy.clone();
        }
    }

    pub fn initialise_games(&mut self) {
        // for i in 0..self.games.len() {
        //     let action: usize =
        //         rand::rng().random_range(0..self.games[i].width * self.games[i].height);
        //     // let action = (self.games[i].height / 2) * self.games[i].width + (self.games[i].width / 2);
        //     self.games[i].generate_bombs(action);
        //     self.games[i].bombs_generated = true;
        //     self.games[i].step(action);
        // }
        for i in 0..self.games.len() {
            self.initialise_game(i);
        }
    }

    pub fn initialise_game(&mut self, idx: usize) {
        let action: usize = rand::rng().random_range(0..self.games[idx].width * self.games[idx].height);
        // let action = (self.games[i].height / 2) * self.games[i].width + (self.games[i].width / 2);
        self.games[idx].generate_bombs(action);
        self.games[idx].bombs_generated = true;
        self.games[idx].step(action);
    }

    pub fn train_step(&mut self) {
        for i in 0..self.games.len() {
            // if !self.games[i].bombs_generated {
            //     let action: usize =
            //         rand::rng().random_range(0..self.games[i].width * self.games[i].height);
            //     // let action = (self.games[i].height / 2) * self.games[i].width + (self.games[i].width / 2);
            //     self.games[i].generate_bombs(action);
            //     self.games[i].bombs_generated = true;
            //     self.games[i].step(action);
            //     return;
            // }

            let obs = self.games[i].to_observation();
            // let obs_tensor = obs_to_tensor(&obs, &self.device);
            let obs_vec = obs_to_vec(&obs);
            let obs_tensor = Tensor::<Backend, 4>::from_floats(
                TensorData::new(
                    obs_vec.clone(),
                    [1, 3, self.games[i].height, self.games[i].width],
                ),
                &self.device,
            );

            let (q_values, _) = self.policy.valid().forward(obs_tensor.inner());

            // NOTE: PPO
            // let logits = logits.squeeze_dim(0);

            let action_size = self.games[i].width * self.games[i].height;
            //
            // let mask_vec = self.games[i].action_mask();
            // let mask_tensor = Tensor::<Backend, 1>::from_data(
            //     TensorData::new(mask_vec.clone(), [action_size]),
            //     &self.device,
            // );
            // let masked_logits = logits + (mask_tensor - 1.0) * 1e9;
            //
            // let uniform = Tensor::<Backend, 1>::random(
            //     [action_size],
            //     burn::tensor::Distribution::Uniform(0.0, 1.0),
            //     &self.device,
            // );
            //
            // let log_probs_for_sample =
            //     log_softmax(masked_logits.clone().unsqueeze_dim::<2>(0), 1).squeeze_dim(0);
            //
            // let gumbel = uniform.log().neg().log().neg();
            // let action = (log_probs_for_sample.clone() + gumbel)
            //     .argmax(0)
            //     .into_scalar() as usize;
            //
            // let log_prob = log_probs_for_sample.narrow(0, action, 1).into_scalar();

            // NOTE: BEFORE PPO
            // let mut logits_vec = logits.clone().into_data().to_vec::<f32>().unwrap();
            // let mask = self.games[i].action_mask();
            //
            // for (l, m) in logits_vec.iter_mut().zip(mask.iter()) {
            //     if *m == 0.0 {
            //         *l = -1e9;
            //     }
            // }

            // let max = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // let exp: Vec<f32> = logits_vec.iter().map(|x| (x - max).exp()).collect();
            // let sum: f32 = exp.iter().sum();
            // let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

            // let prob_sum: f32 = probs.iter().sum();
            // if prob_sum < 0.5 || probs.iter().any(|p| p.is_nan()) {

            let epsilon = (1.0 * (-(self.step_count as f32) / 50000.0).exp()).max(0.05);
            let action = if rand::rng().random::<f32>() < epsilon {
                let valid: Vec<usize> = self.games[i]
                    .action_mask()
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| **m == 1.0)
                    .map(|(i, _)| i)
                    .collect();
                // println!("{}", valid.len());
                if valid.is_empty() {
                    self.episode_count += 1;
                    self.games[i].reset();
                    self.step_count = 0;

                    self.initialise_game(i);

                    self.tx
                        .send(Metric::EpisodeDone {
                            episode: self.episode_count,
                            total_reward: self.last_total_reward,
                            steps: self.last_steps,
                            win: self.last_win,
                            loss: self.last_loss,
                        })
                        .ok();
                    continue;
                }
                valid[rand::rng().random_range(0..valid.len())]
            } else {
                // mask Q values and take argmax
                let mask = self.games[i].action_mask();
                let mask_tensor = Tensor::<InnerBackend, 1>::from_data(
                    TensorData::new(mask, [action_size]),
                    &self.device,
                );
                // let q_values_1d = q_values.reshape([action_size]);
                let masked_q = q_values.squeeze_dim(0) + (mask_tensor - 1.0) * 1e9;
                masked_q.argmax(0).into_scalar() as usize
            };

            let result = self.games[i].step(action);
            self.episode_rewards[i] += result.reward;
            let obs_vec = obs_to_vec(&obs);
            let next_obs = obs_to_vec(&self.games[i].to_observation());

            self.replay_buffer.push(Transition {
                obs: obs_vec,
                action,
                reward: result.reward,
                next_obs,
                done: result.done,
            });

            self.step_count += 1;
            self.games[i].step_count += 1;
            // learn every step once buffer is warm
            if self.replay_buffer.len() >= 256 && self.step_count.is_multiple_of(4) {
                self.train_on_batch(256);
            }

            // NOTE: log_prob for PPO
            // if log_prob.is_nan() || log_prob.is_infinite() {
            //     self.episode_count += 1;
            //     if !self.transitions[i].is_empty() {
            //         self.finish_episode(i);
            //     }
            //     self.games[i].reset();
            //     self.tx
            //         .send(Metric::EpisodeDone {
            //             episode: self.episode_count,
            //             total_reward: self.last_total_reward,
            //             steps: self.last_steps,
            //             win: self.last_win,
            //             loss: self.last_loss,
            //         })
            //         .unwrap();
            //     return;
            // }

            // NOTE: cumulative for BEFORE PPO
            // let r: f32 = rand::random();
            // let mut cumulative = 0.0;
            //
            // let action = probs
            //     .iter()
            //     .enumerate()
            //     .find_map(|(i, p)| {
            //         cumulative += p;
            //         (r < cumulative).then_some(i)
            //     })
            //     .unwrap_or(probs.len() - 1);
            //
            // let result = self.games[i].step(action);
            // self.step_count += 1;

            // self.transitions[i].push(Transition {
            //     observation: obs,
            //     action,
            //     reward: result.reward,
            //     done: result.done,
            // });

            if result.done {
                self.episode_count += 1;
                // if !self.transitions[i].is_empty() {
                //     self.finish_episode(i);
                // }
                self.last_win = result.reward >= 1.0;
                self.last_steps = self.games[i].step_count;
                self.last_total_reward = self.episode_rewards[i];
                self.episode_rewards[i] = 0.0;
                self.games[i].reset();
                // self.step_count = 0;

                self.initialise_game(i);

                self.tx
                    .send(Metric::EpisodeDone {
                        episode: self.episode_count,
                        total_reward: self.last_total_reward,
                        steps: self.last_steps,
                        win: self.last_win,
                        loss: self.last_loss,
                    })
                    .unwrap();
            }
        }
    }
    // fn compute_returns(&self, gamma: f32, game_idx: usize) -> Vec<f32> {
    //     let mut returns = Vec::new();
    //     let mut g = 0.0;
    //     for transition in self.transitions[game_idx].iter().rev() {
    //         g = transition.reward + gamma * g;
    //         returns.push(g);
    //     }
    //     returns.reverse();
    //     returns
    // }
    // fn finish_episode(&mut self, game_idx: usize) {
    //     if self.transitions[game_idx].is_empty() {
    //         return;
    //     }
    //     let returns = self.compute_returns(0.99, game_idx);
    //     let mut returns = returns;
    //     normalise(&mut returns);
    //
    //     let entropy_coeff = 0.05;
    //     let value_coeff = 0.5;
    //     let clip_episilon = 0.2;
    //     let ppo_epochs = 4;
    //
    //     let obs_list: Vec<Tensor<Backend, 4>> = self.transitions[game_idx]
    //         .iter()
    //         .map(|t| obs_to_tensor(&t.observation, &self.device))
    //         .collect();
    //
    //     let obs_batch = Tensor::cat(obs_list, 0);
    //     // println!("obs_batch shae: {:?}", obs_batch.dims());
    //
    //     let old_log_probs_vec: Vec<f32> = self.transitions[game_idx]
    //         .iter()
    //         .map(|t| t.log_prob)
    //         .collect();
    //     let old_log_probs = Tensor::<Backend, 1>::from_data(
    //         TensorData::new(old_log_probs_vec, [self.transitions[game_idx].len()]),
    //         &self.device,
    //     );
    //
    //     let actions: Vec<i32> = self.transitions[game_idx]
    //         .iter()
    //         .map(|t| t.action as i32)
    //         .collect();
    //     let action_tensor = Tensor::<Backend, 1, Int>::from_data(
    //         TensorData::new(actions, [self.transitions[game_idx].len()]),
    //         &self.device,
    //     );
    //
    //     let returns_tensor = Tensor::<Backend, 1>::from_data(
    //         TensorData::new(returns.clone(), [returns.len()]),
    //         &self.device,
    //     );
    //
    //     for _ in 0..ppo_epochs {
    //         let (logits, values) = self.policy.forward(obs_batch.clone());
    //         let values = values.reshape([self.transitions[game_idx].len()]);
    //         let log_probs = log_softmax(logits, 1);
    //
    //         let new_log_probs = log_probs
    //             .clone()
    //             .gather(1, action_tensor.clone().unsqueeze_dim(1))
    //             .squeeze_dim(1);
    //
    //         let entropy: Tensor<Backend, 1> = -(log_probs.clone().exp() * log_probs)
    //             .sum_dim(1)
    //             .squeeze_dim(1);
    //
    //         let advantage = returns_tensor.clone() - values.clone().detach();
    //         // let actor_loss = (selected_log_probs * advantage).mean().neg();
    //         // let critic_loss = (returns_tensor - values).powf_scalar(2.0).sum();
    //         let ratio = (new_log_probs - old_log_probs.clone()).exp();
    //         let unclipped = ratio.clone() * advantage.clone();
    //         let clipped = ratio.clamp(1.0 - clip_episilon, 1.0 + clip_episilon) * advantage;
    //
    //         // let loss =
    //         //     (selected_log_probs * returns_tensor).sum().neg() - entropy.sum() * entropy_coeff;
    //
    //         // let loss: Tensor<Backend, 1> =
    //         //     actor_loss + value_coeff * critic_loss - entropy_coeff * entropy.mean();
    //
    //         let actor_loss = unclipped.min_pair(clipped).mean().neg();
    //         let critic_loss = (returns_tensor.clone() - values).powf_scalar(2.0).mean();
    //
    //         let loss: Tensor<Backend, 1> =
    //             actor_loss + value_coeff * critic_loss - entropy_coeff * entropy.mean();
    //
    //         self.last_loss = loss.clone().into_scalar();
    //
    //         if self.last_loss.is_nan() || self.last_loss.is_infinite() {
    //             println!("NaN/Inf loss detected, resetting policy");
    //             let width = self.games[game_idx].width;
    //             let height = self.games[game_idx].height;
    //             let action_size = width * height;
    //             self.policy = Policy::new(&self.device, height, width, action_size);
    //             self.optim = AdamConfig::new().init();
    //             self.transitions[game_idx].clear();
    //             return;
    //         }
    //
    //         let grads = loss.backward();
    //         let grads = GradientsParams::from_grads(grads, &self.policy);
    //         self.policy = self.optim.step(1e-4, self.policy.clone(), grads);
    //     }
    //
    //     // for ratatui
    //     self.last_win = self.transitions[game_idx]
    //         .last()
    //         .map(|t| t.reward >= 1.0)
    //         .unwrap_or(false);
    //     self.last_total_reward = self.transitions[game_idx].iter().map(|t| t.reward).sum();
    //     self.last_steps = self.transitions[game_idx].len();
    //
    //     self.transitions[game_idx].clear();
    // }
}
// fn normalise(v: &mut [f32]) {
//     let mean = v.iter().sum::<f32>() / v.len() as f32;
//     let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
//     let std = variance.sqrt().max(1e-8);
//     for x in v {
//         *x = ((*x - mean) / std).clamp(-3.0, 3.0);
//     }
// }

fn obs_to_vec(obs: &Observation) -> Vec<f32> {
    let mut v = Vec::with_capacity(obs.hidden.len() * 3);
    v.extend(&obs.hidden);
    v.extend(&obs.revealed);
    v.extend(obs.hints.iter().map(|h| h / 8.0));
    v
}

// fn obs_to_tensor(obs: &Observation, device: &Device<Backend>) -> Tensor<Backend, 4> {
//     let mut input: Vec<f32> = Vec::new();
//     input.extend(&obs.hidden);
//     input.extend(&obs.revealed);
//     input.extend(&obs.hints.iter().map(|h| h / 8.0).collect::<Vec<_>>());
//     // println!("obs height={} width={}", obs.height, obs.width);
//     let data = TensorData::new(input, [1, 3, obs.height, obs.width]);
//     Tensor::<Backend, 4>::from_floats(data, device)
// }

pub fn load_model(tx: Sender<Metric>) -> Model {
    println!("loading model");
    let device = Device::<Backend>::default();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let num_games = 8;
    let height = 5;
    let width = 5;
    // let action_size = 2 * width * height;
    let action_size = width * height;
    let policy = Policy::new(&device, height, width, action_size)
        .load_file(Path::new("model_fcn_dqn"), &recorder, &device)
        .unwrap();
    let target = Policy::new(&device, height, width, action_size)
        .load_file(Path::new("model_fcn_dqn"), &recorder, &device)
        .unwrap();
    Model {
        games: (0..num_games)
            .map(|_| Game::new(height, width, 5))
            .collect(),
        policy,
        target,
        device,
        replay_buffer: ReplayBuffer::new(50_000),
        // transitions: (0..num_games).map(|_| Vec::new()).collect(),
        optim: AdamConfig::new().init(),
        tx,
        target_update_freq: 1000,
        step_count: 0,
        episode_count: 0,
        episode_rewards: (0..num_games).map(|_| 0.0).collect(),
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
        .save_file(Path::new("model_fcn_dqn"), &recorder)
        .unwrap();
    println!("Model saved");
}
