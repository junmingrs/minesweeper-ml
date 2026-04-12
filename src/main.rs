use bevy::{
    DefaultPlugins,
    app::{App, Startup},
    camera::Camera2d,
    ecs::system::Commands,
};

use bevy::prelude::*;

use crate::game::{Cell, Game};

mod game;

const BOMB_COLOR: Color = Color::srgb_u8(255, 50, 50);
const REVEALED_COLOR: Color = Color::srgb_u8(52, 52, 52);
const FLAGGED_COLOR: Color = Color::srgb_u8(100, 100, 200);

#[derive(Component)]
struct CellDisplay {
    x: usize,
    y: usize,
}

#[derive(Component)]
struct CellText;

fn main() {
    let game = Game::new(10, 20, 50);
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(game)
        .add_systems(Startup, setup)
        .add_systems(Update, button_system)
        .add_systems(Update, update_cells)
        .run();
}

fn setup(mut commands: Commands, game: Res<Game>) {
    commands.spawn(Camera2d);
    commands
        .spawn((
            Node {
                display: Display::Grid,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_template_rows: vec![GridTrack::px(50.0); 20],
                grid_template_columns: vec![GridTrack::px(50.0); 10],
                border: UiRect::all(Val::Px(5.0)),
                ..Default::default()
            },
            BackgroundColor(Color::WHITE),
            BorderColor::all(Color::BLACK),
        ))
        .with_children(|builder| {
            for row in game.map.iter() {
                for cell in row.iter() {
                    item_rect(builder, cell);
                }
            }
        });
}

fn item_rect(builder: &mut ChildSpawnerCommands, cell: &Cell) {
    builder
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..Default::default()
            },
            if cell.is_bomb {
                BackgroundColor(BOMB_COLOR.into())
            } else {
                BackgroundColor(cell.color.into())
            },
            Button,
            CellDisplay {
                x: cell.x,
                y: cell.y,
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

fn update_cells(
    cell_query: Query<(&CellDisplay, &mut BackgroundColor, &Children)>,
    mut text_query: Query<&mut Text, With<CellText>>,
    game: Res<Game>,
) {
    for (cell_display, mut background_color, children) in cell_query {
        let cell = game.get_cell(cell_display.x, cell_display.y);
        if cell.revealed {
            println!("revealing {}, {}", cell.x, cell.y);
            background_color.0 = REVEALED_COLOR;
            for child in children.iter() {
                if let Ok(mut text) = text_query.get_mut(child) {
                    text.0 = format!("{}", cell.nearby_bombs);
                }
            }
        }
    }
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor, &CellDisplay, &Children),
        (Changed<Interaction>, With<Button>),
    >,
    mut text_query: Query<&mut Text, With<CellText>>,
    mut game: ResMut<Game>,
) {
    for (interaction, mut background_color, cell_display, children) in &mut interaction_query {
        // TODO: get right click to flag a cell
        if *interaction == Interaction::Pressed {
            let cell = game.get_cell_mut(cell_display.x, cell_display.y);
            if cell.is_bomb {
                continue;
            }
            cell.revealed = true;
            background_color.0 = REVEALED_COLOR;
            for child in children.iter() {
                if let Ok(mut text) = text_query.get_mut(child) {
                    text.0 = format!("{}", cell.nearby_bombs);
                }
            }
            if cell.nearby_bombs == 0 {
                game.reveal_non_zero(cell_display.x, cell_display.y);
            }
        }
    }
}
