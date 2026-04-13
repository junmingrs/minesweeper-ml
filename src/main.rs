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
const REVEALED_PALETTE: [Color; 2] = [Color::srgb_u8(229, 194, 159), Color::srgb_u8(215, 184, 153)];
const FLAGGED_COLOR: Color = Color::srgb_u8(100, 100, 200);
const ONE_COLOR: Color = Color::srgb_u8(25, 118, 210);
const TWO_COLOR: Color = Color::srgb_u8(0, 128, 0);
const THREE_COLOR: Color = Color::srgb_u8(255, 0, 0);
const FOUR_COLOR: Color = Color::srgb_u8(0, 0, 139);
const FIVE_COLOR: Color = Color::srgb_u8(128, 0, 0);
const SIX_COLOR: Color = Color::srgb_u8(0, 255, 255);
const SEVEN_COLOR: Color = Color::srgb_u8(128, 0, 128);
const EIGHT_COLOR: Color = Color::srgb_u8(128, 128, 128);

#[derive(Component)]
struct CellDisplay {
    x: usize,
    y: usize,
}

#[derive(Component)]
struct CellText;

#[derive(Component)]
struct FlagsText;

fn main() {
    let game = Game::new(20, 10, 20);
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(game)
        .add_systems(Startup, setup)
        .add_systems(Update, button_system)
        .add_systems(Update, hover_system)
        .add_systems(Update, update_cells)
        .add_systems(Update, update_flags)
        .add_systems(PostUpdate, check_win)
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
            builder
                .spawn((
                    Node {
                        width: Val::Px(100.),
                        height: Val::Px(50.),
                        position_type: PositionType::Absolute,
                        right: Val::Px(0.),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(BOMB_COLOR.into()),
                    Button,
                ))
                .with_children(|builder| {
                    builder.spawn((
                        Text::new("Restart"),
                        TextFont {
                            font_size: 20.,
                            font_smoothing: bevy::text::FontSmoothing::AntiAliased,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        CellText,
                    ));
                });
            builder
                .spawn((
                    Node {
                        width: Val::Px(100.),
                        height: Val::Px(50.),
                        position_type: PositionType::Absolute,
                        right: Val::Px(0.),
                        top: Val::Px(50.),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(BOMB_COLOR.into()),
                    Button,
                ))
                .with_children(|builder| {
                    builder.spawn((
                        Text::new(format!("{}", game.flags)),
                        TextFont {
                            font_size: 20.,
                            font_smoothing: bevy::text::FontSmoothing::AntiAliased,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        FlagsText,
                    ));
                });
        });
}

fn item_rect(builder: &mut ChildSpawnerCommands, cell: &Cell) {
    builder
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..Default::default()
            },
            BackgroundColor(cell.color.into()),
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
                    font_smoothing: bevy::text::FontSmoothing::AntiAliased,
                    ..default()
                },
                TextColor(Color::WHITE),
                CellText,
            ));
        });
}

fn update_flags(game: Res<Game>, mut query: Query<&mut Text, With<FlagsText>>) {
    for mut text in &mut query {
        text.0 = format!("{}", game.flags);
    }
}

fn update_cells(
    cell_query: Query<(&CellDisplay, &mut BackgroundColor, &Children)>,
    mut text_query: Query<(&mut Text, &mut TextColor), With<CellText>>,
    game: Res<Game>,
) {
    for (cell_display, mut background_color, children) in cell_query {
        // here
        let cell = game.get_cell(cell_display.x, cell_display.y);
        if cell.flagged {
            background_color.0 = FLAGGED_COLOR;
        }
        if cell.revealed {
            background_color.0 = REVEALED_PALETTE[(cell_display.x + cell_display.y) % 2];
            for child in children.iter() {
                if let Ok((mut text, mut text_color)) = text_query.get_mut(child) {
                    text.0 = format!("{}", cell.nearby_bombs);
                    text_color.0 = match cell.nearby_bombs {
                        1 => ONE_COLOR,
                        2 => TWO_COLOR,
                        3 => THREE_COLOR,
                        4 => FOUR_COLOR,
                        5 => FIVE_COLOR,
                        6 => SIX_COLOR,
                        7 => SEVEN_COLOR,
                        8 => EIGHT_COLOR,
                        _ => REVEALED_PALETTE[(cell_display.x + cell_display.y) % 2],
                    }
                }
            }
        } else if !cell.flagged {
            background_color.0 = cell.color;
            for child in children.iter() {
                if let Ok((mut text, mut text_color)) = text_query.get_mut(child) {
                    text.0 = "".to_string();
                    text_color.0 = Color::WHITE;
                }
            }
        }
    }
}

fn hover_system(
    mut interaction_query: Query<(&Interaction, &CellDisplay), With<Button>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut game: ResMut<Game>,
) {
    for (interaction, cell_display) in &mut interaction_query {
        if *interaction == Interaction::Hovered {
            if keyboard_input.just_pressed(KeyCode::KeyF) {
                let cell = game.get_cell_mut(cell_display.x, cell_display.y);
                cell.flagged = !cell.flagged;
                if cell.flagged {
                    game.flags -= 1;
                } else {
                    game.flags += 1;
                }
            }
        }
    }
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &CellDisplay),
        (Changed<Interaction>, With<Button>),
    >,
    mut game: ResMut<Game>,
) {
    for (interaction, cell_display) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            let cell = game.get_cell_mut(cell_display.x, cell_display.y);
            if cell.flagged {
                continue;
            }
            cell.revealed = true;
            if cell.nearby_bombs == 0 && !cell.is_bomb {
                game.reveal_non_zero(cell_display.x, cell_display.y);
            }
        }
    }
}

fn check_win(game: ResMut<Game>, mut commands: Commands) {
    let win = game.check_win();
    if let Some(winlose) = win {
        let height = game.map.len();
        let width = game.map[0].len();
        let game = Game::new(height, width, game.num_bombs);
        commands.remove_resource::<Game>();
        commands.insert_resource(game);
    }
}
