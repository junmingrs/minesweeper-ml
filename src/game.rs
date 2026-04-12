use std::collections::VecDeque;

use bevy::{
    color::Color,
    ecs::{component::Component, resource::Resource},
    math::bool,
};
use rand::RngExt;

const PALETTE: [Color; 2] = [Color::srgb_u8(170, 215, 81), Color::srgb_u8(142, 189, 53)];

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

impl Game {
    pub fn new(width: usize, height: usize, num_bombs: usize) -> Self {
        let mut map: Vec<Vec<Cell>> = Vec::new();
        let mut rng = rand::rng();
        let mut bomb_locations: Vec<(usize, usize)> = Vec::new();
        for _ in 0..num_bombs {
            let x = rng.random_range(0..9);
            let y = rng.random_range(0..19);
            bomb_locations.push((x, y));
        }
        for y in 0..height {
            let mut row = Vec::new();
            for x in 0..width {
                let color = PALETTE[(x + y) % 2];
                row.push(Cell {
                    x,
                    y,
                    is_bomb: bomb_locations.clone().contains(&(x, y)),
                    nearby_bombs: get_nearby_bombs(bomb_locations.clone(), (x, y)),
                    color,
                    revealed: false,
                    flagged: false,
                });
            }
            map.push(row);
        }

        Self { map }
    }
    pub fn get_cell(&self, x: usize, y: usize) -> &Cell {
        &self.map[y][x]
    }
    pub fn get_cell_mut(&mut self, x: usize, y: usize) -> &mut Cell {
        &mut self.map[y][x]
    }
    pub fn reveal_non_zero(&mut self, x: usize, y: usize) {
        let cell = self.get_cell_mut(x, y);
        if cell.nearby_bombs != 0 || cell.is_bomb {
            return;
        }
        cell.revealed = true;
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        queue.push_back((cell.x, cell.y));

        while queue.len() > 0 {
            let (cell_x, cell_y) = queue.pop_front().unwrap();
            let cell = self.get_cell_mut(cell_x, cell_y);
            cell.revealed = true;
            if cell.nearby_bombs != 0 {
                continue;
            }
            for (offset_y, offset_x) in OFFSETS.iter() {
                let pos = calculate_offset(cell_x, cell_y, *offset_x, *offset_y);
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
    }
}

fn calculate_offset(x: usize, y: usize, offset_x: i16, offset_y: i16) -> Option<(usize, usize)> {
    if (x == 0 && offset_x == -1) || (x == 9 && offset_x == 1) {
        return None;
    }
    if (y == 0 && offset_y == -1) || (y == 19 && offset_y == 1) {
        return None;
    }
    let world_x = (x as i16 + offset_x) as usize;
    let world_y = (y as i16 + offset_y) as usize;
    Some((world_x, world_y))
}

fn get_nearby_bombs(bomb_locations: Vec<(usize, usize)>, position: (usize, usize)) -> usize {
    let mut nearby_bombs = 0;
    for (offet_y, offset_x) in OFFSETS.iter() {
        let pos = calculate_offset(position.0, position.1, *offset_x, *offet_y);
        match pos {
            Some((world_x, world_y)) => {
                if bomb_locations.contains(&(world_x, world_y)) {
                    nearby_bombs += 1;
                }
            }
            None => continue,
        }
    }
    nearby_bombs
}
