use std::collections::VecDeque;

use rand::RngExt;

use crate::ml::transition::Transition;

pub struct ReplayBuffer {
    pub transitions: VecDeque<Transition>,
    // pub priorities: VecDeque<f32>,
    pub capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            transitions: VecDeque::with_capacity(capacity),
            // priorities: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() >= self.capacity {
            self.transitions.pop_front();
            // self.priorities.pop_front();
        }
        self.transitions.push_back(transition);
        // self.priorities.push_back(priority);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::rng();
        // let mut indices: Vec<usize> = (0..self.transitions.len()).collect();
        // indices.shuffle(&mut rng);
        // indices[..batch_size].iter()
        //     .map(|&i| &self.transitions[i])
        //     .collect()
        // (0..batch_size)
        //     .map(|_| {
        //         let r = rng.random::<f32>() * total;
        //         self.priorities
        //             .iter()
        //             .enumerate()
        //             .find_map(|(i, &p)| {
        //                 cumsum += p;
        //                 (cumsum >= r).then_some(i)
        //             })
        //             .unwrap_or(self.transitions.len() - 1)
        //     })
        //     .collect()
        (0..batch_size)
            .map(|_| {
                let idx = rng.random_range(0..self.transitions.len());
                &self.transitions[idx]
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}
