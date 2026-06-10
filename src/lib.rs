mod integrators;

use wasm_bindgen::prelude::*;

use glam::{Mat3, Quat, Vec3};
use std::f32;

use crate::integrators::*;

fn run_analysis(
    min_over_range: f32,
    j2_fraction: f32,
    energy_fraction: f32,
    res_t: u32,
    update: fn(Quat, Mat3, Vec3, f32) -> (Quat, Vec3),
) -> AnalysisResults {
    let principal_inv_inertias = min_over_range + Vec3::new(0.0, j2_fraction, 1.0);

    let inv_inertia = Mat3::from_diagonal(principal_inv_inertias);

    let init_theta = 0.5 * (2.0 * energy_fraction - 1.0).acos();

    let mut orientation = Quat::from_scaled_axis(Vec3::Y * init_theta);

    let mut world_momentum = Vec3::Z;
    let dt = 1.0 / (res_t as f32);

    let mut states = vec![State::new(0.0, world_momentum, inv_inertia, orientation)];

    let mut t = 0.0;

    while t < 20.0 * f32::consts::TAU {
        (orientation, world_momentum) = update(orientation, inv_inertia, world_momentum, dt);
        orientation = orientation.normalize();
        t += dt;
        states.push(State::new(t, world_momentum, inv_inertia, orientation));
    }

    AnalysisResults(states)
}

pub struct State {
    time: f32,
    world_momentum: Vec3,
    body_momentum: Vec3,
    momentum_magnitude: f32,
    j_l: f32,
    j_k: f32,
}

impl State {
    pub fn new(time: f32, world_momentum: Vec3, inv_inertia: Mat3, orientation: Quat) -> State {
        let body_momentum = orientation.inverse() * world_momentum;
        let momentum_magnitude2 = world_momentum.length_squared();
        let momentum_magnitude = momentum_magnitude2.sqrt();

        let w = inv_inertia * body_momentum;
        let k = w.cross(body_momentum);

        let j_l = body_momentum.dot(inv_inertia * body_momentum) / momentum_magnitude2;
        let j_k = k.dot(inv_inertia * k) / k.length_squared();

        State {
            time,
            world_momentum: world_momentum / momentum_magnitude,
            body_momentum,
            momentum_magnitude,
            j_l,
            j_k,
        }
    }
}

pub struct AnalysisResults(Vec<State>);

#[wasm_bindgen]
pub fn run(
    min_over_range: f32,
    j2_fraction: f32,
    energy_fraction: f32,
    res_t: u32,
    method: &str,
) -> Vec<f32> {
    let update = match method {
        "explicit" => update_explicit,
        "jolt" => update_jolt,
        "naive_rotation" => update_naive_rotation,
        "proposed_momentum" => update_proposed,
        "proposed_velocity" => update_proposed_velocity,
        "proposed_velocity2" => update_proposed_velocity2,
        "catto_implicit" => update_implicit,
        "buss1" => update_buss1,
        "buss2" => update_buss2,
        "buss2a" => update_buss2a,
        "buss2e" => update_buss2e,
        "buss2ea" => update_buss2ea,
        "buss3a" => update_buss3a,
        _ => update_explicit,
    };
    let results = run_analysis(min_over_range, j2_fraction, energy_fraction, res_t, update);

    results
        .0
        .into_iter()
        .flat_map(|result| {
            let State {
                time,
                world_momentum,
                body_momentum,
                momentum_magnitude,
                j_l,
                j_k,
            } = result;

            [
                time,
                world_momentum.x,
                world_momentum.y,
                world_momentum.z,
                body_momentum.x,
                body_momentum.y,
                body_momentum.z,
                momentum_magnitude,
                j_l,
                j_k,
            ]
        })
        .collect()
}
