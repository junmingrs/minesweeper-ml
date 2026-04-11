use bevy::{
    DefaultPlugins,
    app::{App, Startup},
    camera::Camera2d,
    ecs::system::Commands,
};

use bevy::prelude::*;

const PALETTE: [Color; 2] = [Color::srgb_u8(170, 215, 81), Color::srgb_u8(142, 189, 53)];

#[derive(Component)]
struct GridPosition {
    x: usize,
    y: usize,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, button_system)
        .run();
}

fn setup(mut commands: Commands) {
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
                    item_rect(builder, color, x, y);
                }
            }
        });
}

fn item_rect(builder: &mut ChildSpawnerCommands, color: Color, x: usize, y: usize) {
    builder.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..Default::default()
        },
        BackgroundColor(color.into()),
        Button,
        GridPosition { x, y },
    ));
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &GridPosition),
        (Changed<Interaction>, With<Button>),
    >,
) {
    for (interaction, pos) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            // TODO: add minesweeper logic
            println!("pressed at x: {}, y: {}", pos.x, pos.y);
        }
    }
}
