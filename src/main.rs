use colored::*;
use rand::Rng;
use std::collections::HashSet;
use std::{cmp::Ordering, thread, time::Duration};

const WIDTH: usize = 20;
const HEIGHT: usize = 20;
const ACTIONS: usize = 4;
const OBSTACLE_DENSITY: f64 = 0.2;

type State = (usize, usize);

fn action_to_delta(a: usize) -> (isize, isize) {
    match a {
        0 => (-1, 0),
        1 => (1, 0),
        2 => (0, -1),
        3 => (0, 1),
        _ => (0, 0),
    }
}

fn clamp_pos(x: isize, y: isize) -> (usize, usize) {
    let nx = x.max(0).min((HEIGHT - 1) as isize) as usize;
    let ny = y.max(0).min((WIDTH - 1) as isize) as usize;
    (nx, ny)
}

fn generate_random_walls(start: State, goal: State) -> Vec<State> {
    let mut rng = rand::thread_rng();
    let mut walls = Vec::new();
    let total_cells = WIDTH * HEIGHT;
    let obstacle_count = (total_cells as f64 * OBSTACLE_DENSITY) as usize;

    let mut forbidden: HashSet<State> = HashSet::new();
    forbidden.insert(start);
    forbidden.insert(goal);

    for _ in 0..obstacle_count {
        loop {
            let x = rng.gen_range(0..HEIGHT);
            let y = rng.gen_range(0..WIDTH);
            let pos = (x, y);

            if !forbidden.contains(&pos) {
                walls.push(pos);
                forbidden.insert(pos);
                break;
            }
        }
    }

    walls
}

fn has_path(start: State, goal: State, walls: &[State]) -> bool {
    let wall_set: HashSet<State> = walls.iter().cloned().collect();
    let mut visited = HashSet::new();
    let mut queue = vec![start];

    while let Some(current) = queue.pop() {
        if current == goal {
            return true;
        }

        if visited.contains(&current) {
            continue;
        }
        visited.insert(current);

        for action in 0..ACTIONS {
            let delta = action_to_delta(action);
            let nx = current.0 as isize + delta.0;

            let ny = current.1 as isize + delta.1;
            let next = clamp_pos(nx, ny);

            if next != current && !wall_set.contains(&next) && !visited.contains(&next) {
                queue.insert(0, next);
            }
        }
    }

    false
}

fn is_dead_end(state: State, walls: &[State], visited: &HashSet<State>) -> bool {
    let wall_set: HashSet<State> = walls.iter().cloned().collect();
    let mut valid_moves = 0;

    for action in 0..ACTIONS {
        let delta = action_to_delta(action);
        let nx = state.0 as isize + delta.0;

        let ny = state.1 as isize + delta.1;
        let next = clamp_pos(nx, ny);

        if next != state && !wall_set.contains(&next) && !visited.contains(&next) {
            valid_moves += 1;
        }
    }

    valid_moves <= 1
}

fn main() {
    let episodes = 5000;
    let max_steps = 400;

    let alpha = 0.1;
    let gamma = 0.99;

    let mut epsilon = 1.0;
    let min_epsilon = 0.05;
    let eps_decay = 0.9995;

    let mut rng = rand::thread_rng();

    let start: State = (0, 0);
    let goal: State = loop {
        let gx = rng.gen_range(0..HEIGHT);
        let gy = rng.gen_range(0..WIDTH);
        let candidate = (gx, gy);
        if candidate != start {
            break candidate;
        }
    };

    let walls = loop {
        let candidate_walls = generate_random_walls(start, goal);
        if has_path(start, goal, &candidate_walls) {
            break candidate_walls;
        }
        println!("Regenerating obstacles...");
    };

    println!("Obstacles generated. Count: {}", walls.len());
    println!("Starting Q-Learning training...");

    let mut q = vec![vec![[0.0f64; ACTIONS]; WIDTH]; HEIGHT];

    for ep in 0..episodes {
        let mut state = start;
        let mut visited_in_episode = HashSet::new();

        for _ in 0..max_steps {
            visited_in_episode.insert(state);

            let action = if rng.r#gen::<f64>() < epsilon {
                rng.gen_range(0..ACTIONS)
            } else {
                let row = &q[state.0][state.1];
                let mut best_a = 0usize;
                for a in 1..ACTIONS {
                    if row[a] > row[best_a] {
                        best_a = a;
                    }
                }
                best_a
            };

            let delta = action_to_delta(action);
            let nx = state.0 as isize + delta.0;

            let ny = state.1 as isize + delta.1;
            let next_state = clamp_pos(nx, ny);

            if walls.contains(&next_state) {
                let old_q = q[state.0][state.1][action];
                q[state.0][state.1][action] = old_q + alpha * (-50.0 - old_q);
                continue;
            }

            let mut reward = if next_state == goal {
                200.0
            } else if visited_in_episode.contains(&next_state) {
                -10.0
            } else if is_dead_end(next_state, &walls, &visited_in_episode) {
                -25.0
            } else {
                -1.0
            };

            let distance_to_goal = ((next_state.0 as isize - goal.0 as isize).abs()
                + (next_state.1 as isize - goal.1 as isize).abs())
                as f64;
            let max_distance = (HEIGHT + WIDTH) as f64;
            reward += (max_distance - distance_to_goal) / max_distance * 2.0;

            let old_q = q[state.0][state.1][action];
            let next_max = q[next_state.0][next_state.1]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let new_q = old_q + alpha * (reward + gamma * next_max - old_q);
            q[state.0][state.1][action] = new_q;

            state = next_state;

            if state == goal {
                break;
            }
        }

        epsilon = (epsilon * eps_decay).max(min_epsilon);

        if ep % 500 == 0 {
            println!("Training Episode {} / {}, ε = {:.3}", ep, episodes, epsilon);
        }
    }

    let mut s = start;
    let mut path = vec![s];

    let mut visited_path = HashSet::new();
    let mut stuck_count = 0;

    for _ in 0..500 {
        if s == goal {
            break;
        }

        visited_path.insert(s);

        let mut best_actions: Vec<(usize, f64)> = q[s.0][s.1]
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        best_actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut moved = false;

        for (action, _) in best_actions {
            let delta = action_to_delta(action);
            let nx = s.0 as isize + delta.0;
            let ny = s.1 as isize + delta.1;
            let ns = clamp_pos(nx, ny);

            if !walls.contains(&ns) && ns != s {
                if !visited_path.contains(&ns) || stuck_count > 3 {
                    s = ns;
                    path.push(s);
                    moved = true;
                    stuck_count = 0;
                    break;
                }
            }
        }

        if !moved {
            stuck_count += 1;
            if stuck_count > 5 {
                println!("Agent stuck! Finding new path...");
                if path.len() > 5 {
                    for _ in 0..5 {
                        if let Some(prev_pos) = path.pop() {
                            visited_path.remove(&prev_pos);
                        }
                    }
                    if let Some(&last_pos) = path.last() {
                        s = last_pos;
                    }
                }
                stuck_count = 0;
            }
        }
    }

    for step in 0..path.len() {
        clearscreen::clear().unwrap();

        println!("Step: {} / {}", step + 1, path.len());
        println!("Obstacles: {}", walls.len());
        println!();

        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                if (i, j) == goal {
                    print!(" {} ", "G".bold().green());
                } else if walls.contains(&(i, j)) {
                    print!(" {} ", "#".bold().white());
                } else if (i, j) == path[step] {
                    print!(" {} ", "A".bold().yellow());
                } else if step > 0 && path[..step].contains(&(i, j)) {
                    print!(" {} ", "*".yellow());
                } else if (i, j) == start {
                    print!(" {} ", "S".bold().magenta());
                } else {
                    print!(" {} ", "·".dimmed());
                }
            }
            println!();
        }

        println!();

        thread::sleep(Duration::from_millis(200));
    }

    if path.last() == Some(&goal) {
        println!("{}", "Goal reached successfully!".bold().green());
    } else {
        println!("{}", "Could not reach the goal!".bold().red());
    }
    println!("Path length: {}", path.len());
}
