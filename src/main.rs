use std::{
    sync::{
        Mutex,
        mpsc::{self, Receiver},
    },
    thread::spawn,
};

use bevy::{
    DefaultPlugins,
    app::{App, Startup},
    camera::Camera2d,
};

use bevy::prelude::*;

use crate::{
    game::Cell,
    ml::model::{Model, save_model},
    tui::{Command, Metric, run_tui},
};

mod game;
mod ml;
mod tui;

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
const FLAGGED_BOMB_COLOR: Color = Color::srgb_u8(93, 63, 106);

#[derive(Component)]
struct CellDisplay {
    x: usize,
    y: usize,
}

#[derive(Component)]
struct CellText;

#[derive(Component)]
struct FlagsText;

#[derive(Resource)]
struct CommandReceiver(pub Mutex<Receiver<Command>>);

fn main() {
    let (tx, rx) = mpsc::channel::<Metric>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

    spawn(move || {
        run_tui(rx, cmd_tx).unwrap();
    });
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(Model::new(tx))
        .insert_resource(CommandReceiver(Mutex::new(cmd_rx)))
        .add_systems(Startup, setup)
        // .add_systems(Update, button_system)
        // .add_systems(Update, hover_system)
        .add_systems(Update, train_model)
        .add_systems(Update, update_cells)
        .add_systems(Update, update_flags)
        .add_systems(Update, handle_commands)
        // .add_systems(PostUpdate, check_win)
        .run();
}

fn handle_commands(model: Res<Model>, cmd_rx: Res<CommandReceiver>) {
    if let Ok(cmd) = cmd_rx.0.lock().unwrap().try_recv() {
        match cmd {
            Command::Save => save_model(&model.policy),
        }
    }
}

fn train_model(mut model: ResMut<Model>) {
    model.train_step();
}

fn setup(mut commands: Commands, model: Res<Model>) {
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
            for row in model.game.map.iter() {
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
                        top: Val::Px(50.),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(FLAGGED_COLOR),
                    Button,
                ))
                .with_children(|builder| {
                    builder.spawn((
                        Text::new(format!("Flags: {}", model.game.flags)),
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
            BackgroundColor(cell.color),
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

fn update_flags(model: Res<Model>, mut query: Query<&mut Text, With<FlagsText>>) {
    for mut text in &mut query {
        text.0 = format!("{}", model.game.flags);
    }
}

fn update_cells(
    cell_query: Query<(&CellDisplay, &mut BackgroundColor, &Children)>,
    mut text_query: Query<(&mut Text, &mut TextColor), With<CellText>>,
    model: Res<Model>,
) {
    for (cell_display, mut background_color, children) in cell_query {
        // here
        let cell = model.game.get_cell(cell_display.x, cell_display.y);
        if cell.flagged {
            background_color.0 = FLAGGED_COLOR;
        }
        if cell.revealed {
            if cell.is_bomb { background_color.0 = FLAGGED_BOMB_COLOR; };
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

// fn hover_system(
//     mut interaction_query: Query<(&Interaction, &CellDisplay), With<Button>>,
//     keyboard_input: Res<ButtonInput<KeyCode>>,
//     mut game: ResMut<Game>,
// ) {
//     for (interaction, cell_display) in &mut interaction_query {
//         if *interaction == Interaction::Hovered && keyboard_input.just_pressed(KeyCode::KeyF) {
//             game.apply_action(Action::FlagToggle(cell_display.x, cell_display.y));
//         }
//     }
// }
//
// fn button_system(
//     mut interaction_query: Query<
//         (&Interaction, &CellDisplay),
//         (Changed<Interaction>, With<Button>),
//     >,
//     mut game: ResMut<Game>,
// ) {
//     for (interaction, cell_display) in &mut interaction_query {
//         if *interaction == Interaction::Pressed {
//             game.apply_action(Action::Reveal(cell_display.x, cell_display.y));
//         }
//     }
// }

// fn check_win(mut game: ResMut<Game>) {
//     let win = game.check_win();
//     if win.is_some() {
//         game.reset();
//     }
// }
