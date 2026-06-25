use std::collections::VecDeque;

use rand::seq::SliceRandom;

use crate::ml::transition::Transition;

pub struct ReplayBuffer {
    pub transitions: VecDeque<Transition>,
    pub capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            transitions: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() >= self.capacity {
            self.transitions.pop_front();
        }
        self.transitions.push_back(transition);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..self.transitions.len()).collect();
        indices.shuffle(&mut rng);
        indices[..batch_size].iter()
            .map(|&i| &self.transitions[i])
            .collect()
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}
