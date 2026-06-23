use crate::ml::env::Observation;

pub struct Transition {
    pub observation: Observation,
    pub action: usize,
    pub reward: f32,
    pub done: bool,
    pub log_prob: f32,
}
