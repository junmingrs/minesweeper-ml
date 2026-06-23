use crate::game::{Action, ActionOutcome, Game};

pub struct Observation {
    pub hidden: Vec<f32>,
    pub revealed: Vec<f32>,
    pub flagged: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

pub struct StepResult {
    pub observation: Observation,
    pub reward: f32,
    pub done: bool,
}

pub trait Environment {
    fn decode(&self, action: usize) -> Action;
    fn reset(&mut self) -> Observation;
    fn step(&mut self, action: usize) -> StepResult;
    fn action_mask(&self) -> Vec<f32>;
}

impl Observation {
    pub fn platten(&self) -> Vec<f32> {
        let mut v = Vec::new();

        v.extend(self.hidden.iter().copied());
        v.extend(self.revealed.iter().copied());
        v.extend(self.flagged.iter().copied());
        v
    }
}

impl Environment for Game {
    fn decode(&self, action: usize) -> Action {
        let board_size = self.width * self.height;
        let idx = action % board_size;
        let x = idx % self.width;
        let y = idx / self.width;
        if action < board_size {
            Action::Reveal(x, y)
        } else {
            Action::FlagToggle(x, y)
        }
    }
    fn reset(&mut self) -> Observation {
        *self = Game::new(self.height, self.width, self.num_bombs);
        self.to_observation()
    }
    fn step(&mut self, action: usize) -> StepResult {
        let action = self.decode(action);
        // NOTE: include Action::FlagToggle later
        let outcome = self.apply_action(action);

        let (reward, done) = match outcome {
            ActionOutcome::RevealCell(n) => (0.05 * n, false),
            ActionOutcome::FlagPlaced => (0.02, false),
            ActionOutcome::FlagRemoved => (-0.1, false),
            ActionOutcome::Invalid => (-0.5, false),
            ActionOutcome::HitBomb => (-1.0, true),
            ActionOutcome::Win => (2.0, true),
        };

        StepResult {
            observation: self.to_observation(),
            reward,
            done,
        }
    }
    fn action_mask(&self) -> Vec<f32> {
        let board_size = self.width * self.height;
        let mut mask = vec![1.0; board_size * 2];

        for y in 0..self.height {
            for x in 0..self.width {
                let i = y * self.width + x;
                let cell = self.get_cell(x, y);

                if cell.revealed || cell.flagged {
                    mask[i] = 0.0;
                }

                let flag_idx = i + board_size;
                if cell.revealed {
                    mask[flag_idx] = 0.0;
                }
            }
        }

        mask
    }
}
