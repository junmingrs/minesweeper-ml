use std::collections::VecDeque;

use bevy::{
    color::Color,
    ecs::{component::Component, resource::Resource},
    math::bool,
};
use rand::RngExt;

use crate::ml::env::Observation;

const PALETTE: [Color; 2] = [Color::srgb_u8(170, 215, 81), Color::srgb_u8(142, 189, 53)];
const BOMB_COLOR: Color = Color::srgb_u8(255, 50, 50);

#[derive(Component)]
pub struct Cell {
    pub x: usize,
    pub y: usize,
    pub is_bomb: bool,
    pub nearby_bombs: usize,
    pub color: Color,
    pub revealed: bool,
    pub flagged: bool,
}

#[derive(Resource)]
pub struct Game {
    pub map: Vec<Vec<Cell>>,
    pub height: usize,
    pub width: usize,
    pub num_bombs: usize,
    // pub flags: usize,
    pub bombs_generated: bool,
}

const OFFSETS: [(i16, i16); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

pub enum Action {
    Reveal(usize, usize),
    // FlagToggle(usize, usize),
}

pub enum ActionOutcome {
    RevealCell(f32),
    HitBomb,
    // FlagPlaced,
    // FlagRemoved,
    Invalid,
    Win,
}

impl Game {
    pub fn new(height: usize, width: usize, num_bombs: usize) -> Self {
        let mut map: Vec<Vec<Cell>> = Vec::new();
        for y in 0..height {
            let mut row = Vec::new();
            for x in 0..width {
                // let color: Color;
                // let is_bomb = bomb_locations.clone().contains(&(x, y));
                // if is_bomb {
                //     color = BOMB_COLOR;
                // } else {
                //     color = PALETTE[(x + y) % 2];
                // }
                row.push(Cell {
                    x,
                    y,
                    is_bomb: false,
                    // nearby_bombs: get_nearby_bombs(bomb_locations.clone(), (x, y)),
                    nearby_bombs: 0,
                    color: PALETTE[(x + y) % 2],
                    revealed: false,
                    flagged: false,
                });
            }
            map.push(row);
        }

        Self {
            map,
            height,
            width,
            // num_bombs: bomb_locations.len(),
            num_bombs,
            // flags: bomb_locations.len(),
            bombs_generated: false,
        }
    }
    pub fn generate_bombs(&mut self, action: usize) {
        let (safe_x, safe_y) = (action % self.width, action / self.width);
        let mut rng = rand::rng();
        let mut i = 0;
        while i < self.num_bombs {
            let x = rng.random_range(0..self.width);
            let y = rng.random_range(0..self.height);
            let too_close = (x as i32 - safe_x as i32).abs() <= 1 && (y as i32 - safe_y as i32).abs() <= 1;
            let cell = self.get_cell_mut(x, y);
            // if !(cell.is_bomb || x == safe_x && y == safe_y) {
            if !cell.is_bomb && !too_close {
                cell.is_bomb = true;
                cell.color = BOMB_COLOR;
                i += 1;
            }
        }
        self.recalculate_hints();
    }
    pub fn recalculate_hints(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                if !self.map[y][x].is_bomb {
                    self.map[y][x].nearby_bombs = self.get_nearby_bombs(x, y);
                }
            }
        }
    }
    // pub fn reset(&self) -> Game {
    //     Game::new(self.map.len(), self.map[0].len(), self.num_bombs)
    // }
    pub fn get_cell(&self, x: usize, y: usize) -> &Cell {
        &self.map[y][x]
    }
    pub fn get_cell_mut(&mut self, x: usize, y: usize) -> &mut Cell {
        &mut self.map[y][x]
    }
    pub fn reveal_non_zero(&mut self, x: usize, y: usize) -> f32 {
        let cell = self.get_cell_mut(x, y);
        if cell.nearby_bombs != 0 || cell.is_bomb {
            return 0.0;
        }
        let mut cells_revealed = 1_f32;
        cell.revealed = true;
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        queue.push_back((cell.x, cell.y));

        while !queue.is_empty() {
            let (cell_x, cell_y) = queue.pop_front().unwrap();
            let cell = self.get_cell_mut(cell_x, cell_y);
            cell.revealed = true;
            cells_revealed += 1.0;
            if cell.nearby_bombs != 0 {
                continue;
            }
            for (offset_y, offset_x) in OFFSETS.iter() {
                let pos = self.calculate_offset(cell_x, cell_y, *offset_x, *offset_y);
                match pos {
                    Some((world_x, world_y)) => {
                        let cell = self.get_cell(world_x, world_y);
                        if !cell.revealed && !cell.is_bomb {
                            queue.push_back((world_x, world_y));
                        }
                    }
                    None => continue,
                }
            }
        }
        cells_revealed
    }
    pub fn check_win(&self) -> Option<bool> {
        let mut safe_opened = 0;
        let num_safe_cells = (self.map.len() * self.map[0].len()) - self.num_bombs;
        let mut not_safe_flagged = 0;
        for row in self.map.iter() {
            for cell in row.iter() {
                if cell.is_bomb && cell.revealed {
                    return Some(false);
                }
                if !cell.is_bomb && cell.revealed {
                    safe_opened += 1;
                }
                if cell.is_bomb && cell.flagged {
                    not_safe_flagged += 1;
                }
            }
        }
        if (safe_opened == num_safe_cells) || (not_safe_flagged == self.num_bombs) {
            return Some(true);
        }
        None
    }
    // pub fn apply_action(&mut self, action: Action) -> ActionOutcome {
    pub fn apply_action(&mut self, action: Action) -> ActionOutcome {
        match action {
            Action::Reveal(x, y) => {
                let cell = self.get_cell_mut(x, y);
                let mut cells_revealed = 0_f32;
                if cell.flagged || cell.revealed {
                    return ActionOutcome::Invalid;
                }
                cell.revealed = true;
                if cell.is_bomb {
                    return ActionOutcome::HitBomb;
                }
                if cell.nearby_bombs == 0 {
                    cells_revealed = self.reveal_non_zero(x, y);
                }
                if let Some(win) = self.check_win()
                    && win
                {
                    return ActionOutcome::Win;
                }
                ActionOutcome::RevealCell(cells_revealed)
            } // Action::FlagToggle(x, y) => {
              //     let no_flags = self.flags == 0;
              //     let delta = {
              //         let cell = self.get_cell_mut(x, y);
              //         if !cell.flagged && !no_flags {
              //             cell.flagged = true;
              //             -1
              //         } else if cell.flagged {
              //             cell.flagged = false;
              //             1
              //         } else {
              //             0
              //         }
              //     };
              //     match delta {
              //         -1 => {
              //             self.flags -= 1;
              //             ActionOutcome::FlagPlaced
              //         }
              //         1 => {
              //             self.flags += 1;
              //             ActionOutcome::FlagRemoved
              //         }
              //         _ => {
              //             ActionOutcome::Invalid
              //         }
              //     }
              // }
        }
    }
    pub fn to_observation(&self) -> Observation {
        let width = self.map[0].len();
        let height = self.map.len();
        let mut hidden = Vec::with_capacity(width * height);
        let mut revealed = Vec::with_capacity(width * height);
        // let mut flagged = Vec::with_capacity(width * height);
        let mut hints = Vec::with_capacity(width * height);

        for row in &self.map {
            for cell in row {
                hidden.push(if cell.revealed { 0.0 } else { 1.0 });
                revealed.push(if cell.revealed { 1.0 } else { 0.0 });
                hints.push(if cell.revealed {
                    cell.nearby_bombs as f32
                } else {
                    -1.0
                });
                // flagged.push(if cell.flagged { 1.0 } else { 0.0 });
            }
        }
        Observation {
            hidden,
            revealed,
            // flagged,
            hints,
            height,
            width,
        }
    }
    fn get_nearby_bombs(&self, x: usize, y: usize) -> usize {
        let mut nearby_bombs = 0;
        for (offet_y, offset_x) in OFFSETS.iter() {
            let pos = self.calculate_offset(x, y, *offset_x, *offet_y);
            match pos {
                Some((world_x, world_y)) => {
                    if self.get_cell(world_x, world_y).is_bomb {
                        nearby_bombs += 1;
                    }
                }
                None => continue,
            }
        }
        nearby_bombs
    }
    fn calculate_offset(
        &self,
        x: usize,
        y: usize,
        offset_x: i16,
        offset_y: i16,
    ) -> Option<(usize, usize)> {
        if (x == 0 && offset_x == -1) || (x == self.width - 1 && offset_x == 1) {
            return None;
        }
        if (y == 0 && offset_y == -1) || (y == self.height - 1 && offset_y == 1) {
            return None;
        }
        let world_x = (x as i16 + offset_x) as usize;
        let world_y = (y as i16 + offset_y) as usize;
        Some((world_x, world_y))
    }
}
