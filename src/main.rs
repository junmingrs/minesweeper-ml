use std::{
    path::Path,
    sync::mpsc::{self},
    thread::spawn,
};

use crate::{
    // bevy::run_bevy,
    ml::model::{Model, load_model, save_model},
    tui::{Command, Metric, run_tui},
};

// mod bevy;
mod game;
mod ml;
mod tui;

fn main() {
    let (tx, rx) = mpsc::channel::<Metric>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

    let mut model: Model = if Path::new("model_fcn_dqn.mpk").exists() {
        load_model(tx.clone())
    } else {
        Model::new(tx)
    };
    // spawn(move || {
    //     run_bevy(model, cmd_rx);
    // });

    spawn(move || {
        run_tui(rx, cmd_tx).unwrap();
    });

    // run_bevy(model, cmd_rx);

    model.initialise_games();
    model.warmup(10_000);

    loop {
        model.train_step();
        match cmd_rx.try_recv() {
            Ok(_) => {
                save_model(&model.policy, model.step_count);
            }
            Err(_) => {
                // println!("could not save model!");
            }
        }
    }
}
