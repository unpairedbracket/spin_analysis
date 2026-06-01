use glam::{Mat3, Quat, Vec3};

pub fn update_explicit(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);
    let body_l1 = body_l0 - k * dt;
    let world_l1 = q * body_l1;
    let world_w1 = q * (J * body_l1);

    let q_step = Quat::from_scaled_axis(world_w1 * dt);
    let q1 = q_step * q;
    let world_l_out = q_step * world_l1;
    (q1, world_l_out)
}

pub fn update_jolt(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);
    let body_l1 = body_l0 - k * dt;
    let body_l1 = body_l1 * (body_l0.length() / body_l1.length());
    let world_l1 = q * body_l1;
    let world_w1 = q * (J * body_l1);

    let q_step = Quat::from_scaled_axis(world_w1 * dt);
    let q1 = q_step * q;
    let world_l_out = q_step * world_l1;
    (q1, world_l_out)
}

pub fn update_naive_rotation(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;

    let omega_world = q * body_w;
    let q_omega = Quat::from_scaled_axis(omega_world * dt);

    let q1 = q_omega * q;

    (q1, world_momentum)
}

pub fn update_proposed(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);
    let j_k = k.dot(J * k) / k.length_squared();
    let omega = body_w - j_k * body_l0;
    let omega_world = q * omega;
    let q_omega = Quat::from_scaled_axis(omega_world * dt);
    let q_l = Quat::from_scaled_axis(j_k * world_momentum * dt);

    let q1 = q_l * q_omega * q;

    (q1, world_momentum)
}

pub fn update_proposed_velocity(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);
    let j_k = k.dot(J * k) / k.length_squared();
    let omega = body_w - j_k * body_l0;

    let q_omega = Quat::from_scaled_axis(-omega * dt);

    let body_l1 = q_omega * body_l0;
    let body_w1 = J * body_l1;

    let world_l1 = q * body_l1;
    let world_w1 = q * body_w1;

    let q_step = Quat::from_scaled_axis(world_w1 * dt);

    let q1 = q_step * q;
    let world_l_out = q_step * world_l1;

    (q1, world_l_out)
}
