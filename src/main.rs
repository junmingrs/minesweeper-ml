use core::num;
use std::{backtrace, collections::HashMap};

use bevy::{
    DefaultPlugins,
    app::{App, Startup},
    camera::Camera2d,
    ecs::system::Commands,
};

use bevy::prelude::*;
use rand::RngExt;

const PALETTE: [Color; 2] = [Color::srgb_u8(170, 215, 81), Color::srgb_u8(142, 189, 53)];
const BOMB_COLOR: Color = Color::srgb_u8(255, 50, 50);

#[derive(Component)]
struct Cell {
    x: usize,
    y: usize,
    nearby_bombs: usize,
    bomb: bool,
}

#[derive(Component)]
struct CellText;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, button_system)
        .run();
}

fn setup(mut commands: Commands) {
    let mut rng = rand::rng();
    let num_bombs = 20_usize;
    let mut bomb_locations: Vec<(usize, usize)> = Vec::new();
    for _ in 0..num_bombs {
        let x = rng.random_range(0..9);
        let y = rng.random_range(0..19);
        bomb_locations.push((x, y));
    }
    commands.spawn(Camera2d);
    commands
        .spawn((
            Node {
                display: Display::Grid,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_template_rows: vec![GridTrack::px(50.0); 20],
                grid_template_columns: vec![GridTrack::px(50.0); 9],
                border: UiRect::all(Val::Px(5.0)),
                ..Default::default()
            },
            BackgroundColor(Color::WHITE),
            BorderColor::all(Color::BLACK),
        ))
        .with_children(|builder| {
            for y in 0..20 {
                for x in 0..9 {
                    let color = PALETTE[(x + y) % 2];
                    item_rect(
                        builder,
                        color,
                        x,
                        y,
                        bomb_locations.clone().contains(&(x, y)),
                        get_nearby_bombs(bomb_locations.clone(), (x, y)),
                    );
                }
            }
        });
}

fn item_rect(
    builder: &mut ChildSpawnerCommands,
    color: Color,
    x: usize,
    y: usize,
    bomb: bool,
    nearby_bombs: usize,
) {
    builder
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..Default::default()
            },
            if bomb {
                BackgroundColor(BOMB_COLOR.into())
            } else {
                BackgroundColor(color.into())
            },
            Button,
            Cell {
                x,
                y,
                nearby_bombs,
                bomb,
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont {
                    font_size: 30.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                CellText,
            ));
        });
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor, &Cell, &Children),
        (Changed<Interaction>, With<Button>),
    >,
    mut text_query: Query<&mut Text, With<CellText>>,
) {
    for (interaction, mut backgroundcolor, cell, children) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            // TODO: add minesweeper logic
            // left with algo to open nearby 0 cells
            // if cell is zero, reveal all neighbours
            println!("pressed at x: {}, y: {}", cell.x, cell.y);
            println!("bomb: {}", cell.bomb);
            println!("nearby bombs: {}", cell.nearby_bombs);
            if cell.bomb {
                continue;
            }
            backgroundcolor.0 = Color::srgb_u8(52, 52, 52);
            for child in children.iter() {
                if let Ok(mut text) = text_query.get_mut(child) {
                    text.0 = format!("{}", cell.nearby_bombs);
                }
            }
        }
    }
}

fn get_nearby_bombs(bomb_locations: Vec<(usize, usize)>, position: (usize, usize)) -> usize {
    let offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let mut nearby_bombs = 0;
    for (y, x) in offsets.iter() {
        if (position.0 == 0 && *x == -1) || (position.0 == 9 && *x == 1) {
            continue;
        }
        if (position.1 == 0 && *y == -1) || (position.1 == 19 && *y == 1) {
            continue;
        }
        let a = (position.0 as i16 + x) as usize;
        let b = (position.1 as i16 + y) as usize;
        if bomb_locations.contains(&(a, b)) {
            nearby_bombs += 1;
        }
    }
    nearby_bombs
}
