pub struct Transition {
    // pub observation: Observation,
    pub obs: Vec<f32>,
    pub next_obs: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    // pub next_observation: Observation,
    pub done: bool,
    // pub log_prob: f32,
}
