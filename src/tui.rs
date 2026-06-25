use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use ratatui::crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, Paragraph},
};

// ── Metric event sent from training thread ──────────────────────────────────

pub enum Metric {
    EpisodeDone {
        episode: usize,
        total_reward: f32,
        steps: usize,
        win: bool,
        loss: f32,
    },
}

pub enum Command {
    Save,
}

// ── App state ────────────────────────────────────────────────────────────────

struct AppTUI {
    episode: usize,
    wins: usize,
    losses: usize,
    total_steps: usize,
    recent_rewards: Vec<f64>,   // last 200 episodes
    recent_losses: Vec<f64>,    // last 200 episodes
    win_rate_history: Vec<f64>, // last 200 episodes (0.0–1.0)
    log: Vec<String>,           // last N lines
    start: Instant,
}

impl AppTUI {
    pub fn new() -> Self {
        AppTUI {
            episode: 0,
            wins: 0,
            losses: 0,
            total_steps: 0,
            recent_rewards: Vec::new(),
            recent_losses: Vec::new(),
            win_rate_history: Vec::new(),
            log: Vec::new(),
            start: Instant::now(),
        }
    }

    fn push(&mut self, metric: Metric) {
        match metric {
            Metric::EpisodeDone {
                episode,
                total_reward,
                steps,
                win,
                loss,
            } => {
                self.episode = episode;
                self.total_steps += steps;
                if win {
                    self.wins += 1;
                } else {
                    self.losses += 1;
                }

                push_capped(&mut self.recent_rewards, total_reward as f64, 200);
                push_capped(&mut self.recent_losses, loss as f64, 200);

                let wr = self.wins as f64 / self.episode.max(1) as f64;
                push_capped(&mut self.win_rate_history, wr, 200);

                let status = if win { "WIN " } else { "LOSS" };
                self.log.push(format!(
                    "ep {:>6}  {}  reward {:>7.2}  steps {:>4}  loss {:>8.4}",
                    episode, status, total_reward, steps, loss
                ));
                if self.log.len() > 200 {
                    self.log.remove(0);
                }
            }
        }
    }

    fn win_rate(&self) -> f64 {
        if self.episode == 0 {
            return 0.0;
        }
        self.wins as f64 / self.episode as f64
    }

    fn avg_reward(&self) -> f64 {
        if self.recent_rewards.is_empty() {
            return 0.0;
        }
        self.recent_rewards.iter().sum::<f64>() / self.recent_rewards.len() as f64
    }

    fn avg_loss(&self) -> f64 {
        if self.recent_losses.is_empty() {
            return 0.0;
        }
        self.recent_losses.iter().sum::<f64>() / self.recent_losses.len() as f64
    }

    fn elapsed(&self) -> String {
        let s = self.start.elapsed().as_secs();
        format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
    }
}

fn push_capped(v: &mut Vec<f64>, val: f64, cap: usize) {
    v.push(val);
    if v.len() > cap {
        v.remove(0);
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

pub fn run_tui(rx: Receiver<Metric>, cmd_tx: Sender<Command>) -> std::io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = AppTUI::new();
    let tick = Duration::from_millis(200);

    loop {
        // drain all pending metrics
        loop {
            match rx.try_recv() {
                Ok(m) => app.push(m),
                Err(_) => break,
            }
        }

        terminal.draw(|f| ui(f, &app))?;

        // keyboard: q or Esc to quit
        if event::poll(tick)? {
            if let Event::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char('s') => {
                        cmd_tx.send(Command::Save).ok();
                        app.log.push("--- save requested ---".to_string());
                    }
                    KeyCode::Char('q') => break,
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

// ── UI layout ────────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &AppTUI) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title bar
            Constraint::Length(5), // stat cards
            Constraint::Min(10),   // charts + log
        ])
        .split(f.area());

    draw_title(f, root[0], app);
    draw_stats(f, root[1], app);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(root[2]);

    draw_charts(f, bottom[0], app);
    draw_log(f, bottom[1], app);
}

fn draw_title(f: &mut Frame, area: Rect, app: &AppTUI) {
    let text = vec![Line::from(vec![
        Span::styled(
            "⛏  Minesweeper RL",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("   "),
        Span::styled(
            format!("elapsed {}", app.elapsed()),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("   "),
        Span::styled("q / Esc to quit", Style::default().fg(Color::DarkGray)),
    ])];
    let p = Paragraph::new(text).block(Block::default().borders(Borders::ALL));
    f.render_widget(p, area);
}

fn draw_stats(f: &mut Frame, area: Rect, app: &AppTUI) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 6),
            Constraint::Ratio(1, 6),
            Constraint::Ratio(1, 6),
            Constraint::Ratio(1, 6),
            Constraint::Ratio(1, 6),
            Constraint::Ratio(1, 6),
        ])
        .split(area);

    stat_card(
        f,
        cols[0],
        "Episodes",
        &app.episode.to_string(),
        Color::White,
    );
    stat_card(
        f,
        cols[1],
        "Win rate",
        &format!("{:.1}%", app.win_rate() * 100.0),
        Color::Green,
    );
    stat_card(
        f,
        cols[2],
        "Win rate / 200 ep",
        &format!("{:.1}%", app.win_rate_history.iter().sum::<f64>() / 2.0),
        Color::LightGreen,
    );
    stat_card(
        f,
        cols[3],
        "Avg reward",
        &format!("{:.3}", app.avg_reward()),
        Color::Yellow,
    );
    stat_card(
        f,
        cols[4],
        "Avg loss",
        &format!("{:.4}", app.avg_loss()),
        Color::Red,
    );
    stat_card(
        f,
        cols[5],
        "Total steps",
        &app.total_steps.to_string(),
        Color::Cyan,
    );
}

fn stat_card(f: &mut Frame, area: Rect, label: &str, value: &str, color: Color) {
    let text = vec![
        Line::from(Span::styled(
            value,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(label, Style::default().fg(Color::DarkGray))),
    ];
    let p = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL))
        .alignment(ratatui::layout::Alignment::Center);
    f.render_widget(p, area);
}

fn draw_charts(f: &mut Frame, area: Rect, app: &AppTUI) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(area);

    draw_line_chart(
        f,
        rows[0],
        &app.win_rate_history,
        "Win rate",
        Color::Green,
        0.0,
        1.0,
    );
    draw_line_chart(
        f,
        rows[1],
        &app.recent_rewards,
        "Reward",
        Color::Yellow,
        reward_min(app),
        reward_max(app),
    );
    draw_line_chart(
        f,
        rows[2],
        &app.recent_losses,
        "Loss",
        Color::Red,
        0.0,
        loss_max(app),
    );
}

fn draw_line_chart(
    f: &mut Frame,
    area: Rect,
    data: &[f64],
    title: &str,
    color: Color,
    y_min: f64,
    y_max: f64,
) {
    let points: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();

    let datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(color))
            .data(&points),
    ];

    let x_max = (data.len().max(1) - 1) as f64;
    let chart = Chart::new(datasets)
        .block(Block::default().title(title).borders(Borders::ALL))
        .x_axis(Axis::default().bounds([0.0, x_max.max(1.0)]))
        .y_axis(
            Axis::default()
                .bounds([y_min, y_max.max(y_min + 0.01)])
                .labels(vec![
                    Span::raw(format!("{:.2}", y_min)),
                    Span::raw(format!("{:.2}", (y_min + y_max) / 2.0)),
                    Span::raw(format!("{:.2}", y_max)),
                ]),
        );

    f.render_widget(chart, area);
}

fn draw_log(f: &mut Frame, area: Rect, app: &AppTUI) {
    let items: Vec<ListItem> = app
        .log
        .iter()
        .rev()
        .take(area.height as usize)
        .map(|l| {
            let color = if l.contains("WIN") {
                Color::Green
            } else {
                Color::Red
            };
            ListItem::new(Line::from(Span::styled(l, Style::default().fg(color))))
        })
        .collect();

    let list = List::new(items).block(Block::default().title("Episode log").borders(Borders::ALL));
    f.render_widget(list, area);
}

// ── helpers for chart bounds ─────────────────────────────────────────────────

fn reward_min(app: &AppTUI) -> f64 {
    app.recent_rewards
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .min(0.0)
}
fn reward_max(app: &AppTUI) -> f64 {
    app.recent_rewards
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0)
}
fn loss_max(app: &AppTUI) -> f64 {
    app.recent_losses
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0)
}
