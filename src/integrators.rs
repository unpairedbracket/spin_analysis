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
    let k2 = k.length_squared();

    let q_step = if k2 > 1e-8 {
        let j_k = k.dot(J * k) / k2;
        let omega = body_w - j_k * body_l0;
        let omega_world = q * omega;
        let q_omega = Quat::from_scaled_axis(omega_world * dt);
        let q_l = Quat::from_scaled_axis(j_k * world_momentum * dt);

        q_l * q_omega
    } else {
        let omega_world = q * body_w;
        let q_omega = Quat::from_scaled_axis(omega_world * dt);

        q_omega
    };

    let q1 = q_step * q;

    (q1, world_momentum)
}

pub fn update_proposed_velocity(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);

    let k2 = k.length_squared();

    let body_l1 = if k2 > 1e-8 {
        let j_k = k.dot(J * k) / k2;
        let omega = body_w - j_k * body_l0;
        let q_omega = Quat::from_scaled_axis(-omega * dt);
        q_omega * body_l0
    } else {
        body_l0
    };

    let body_w1 = J * body_l1;
    let world_l1 = q * body_l1;
    let world_w1 = q * body_w1;
    let q_step = Quat::from_scaled_axis(world_w1 * dt);
    let q1 = q_step * q;
    let world_l_out = q_step * world_l1;

    (q1, world_l_out)
}

pub fn update_buss1(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);

    let body_wbar = body_w;

    let wbar = q * body_wbar;
    let q_omega = Quat::from_scaled_axis(wbar * dt);

    let q1 = q_omega * q;

    (q1, world_momentum)
}
pub fn update_buss2(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);

    let wdot = -J * k;

    let body_wbar = body_w + wdot * dt / 2.0;

    let wbar = q * body_wbar;
    let q_omega = Quat::from_scaled_axis(wbar * dt);

    let q1 = q_omega * q;

    (q1, world_momentum)
}

pub fn update_buss2a(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);

    let j_k = k.dot(J * k) / k.length_squared();
    let wdot = -j_k * k;

    let body_wbar = body_w + dt / 2.0 * wdot;

    let wbar = q * body_wbar;
    let q_omega = Quat::from_scaled_axis(wbar * dt);

    let q1 = q_omega * q;

    (q1, world_momentum)
}

pub fn update_buss2e(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);

    let wdot = -J * k;

    let body_wbar = body_w + wdot * dt / 2.0 + wdot.cross(body_w) * dt.powi(2) / 12.0;

    let wbar = q * body_wbar;
    let q_omega = Quat::from_scaled_axis(wbar * dt);

    let q1 = q_omega * q;

    (q1, world_momentum)
}

pub fn update_buss2ea(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);

    let j_k = k.dot(J * k) / k.length_squared();
    let wdot = -j_k * k;

    let body_wbar = body_w + dt / 2.0 * wdot + wdot.cross(body_w) * dt.powi(2) / 12.0;

    let wbar = q * body_wbar;
    let q_omega = Quat::from_scaled_axis(wbar * dt);

    let q1 = q_omega * q;

    (q1, world_momentum)
}

pub fn update_buss3a(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let k = body_w.cross(body_l0);

    let j_k = k.dot(J * k) / k.length_squared();
    let wdot = -j_k * k;
    let j_l = body_l0.dot(J * body_l0) / body_l0.length_squared();

    let wddot = -(body_w.cross(wdot)
        + 2.0 * j_k.powi(2) * body_l0.length_squared() * (body_w - j_l * body_l0));

    let body_wbar = body_w + dt / 2.0 * wdot + wddot * dt.powi(2) / 12.0;

    let wbar = q * body_wbar;
    let q_omega = Quat::from_scaled_axis(wbar * dt);

    let q1 = q_omega * q;

    (q1, world_momentum)
}

pub fn skew_symmetric_mat3(v: Vec3) -> Mat3 {
    Mat3::from_cols_array(&[0.0, v.z, -v.y, -v.z, 0.0, v.x, v.y, -v.x, 0.0])
}

pub fn update_implicit(q: Quat, J: Mat3, world_momentum: Vec3, dt: f32) -> (Quat, Vec3) {
    let body_l0 = q.inverse() * world_momentum;
    let body_w = J * body_l0;
    let local_inertia = J.inverse();

    let jacobian = local_inertia
        + dt * (skew_symmetric_mat3(body_w) * local_inertia - skew_symmetric_mat3(body_l0));

    let k = body_w.cross(body_l0);

    let ang_acc = -jacobian.inverse_or_zero() * k;

    let body_w1 = body_w + ang_acc * dt;
    let body_l1 = local_inertia * body_w1;

    let world_w1 = q * body_w1;
    let world_l1 = q * body_l1;

    let q_step = Quat::from_scaled_axis(world_w1 * dt);
    let q1 = q_step * q;
    let world_l_out = q_step * world_l1;
    (q1, world_l_out)
}
