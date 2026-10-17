//! Helpers shared by the integration tests.
#![allow(dead_code)] // each test crate uses a different subset

use eg2cpu::Model;
use serde_json::Value;

/// The default checkpoint, or None (with a SKIPPED line on stderr) when there is none, as in CI.
pub fn model_or_skip(test: &str) -> Option<Model> {
    match Model::load_default() {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("SKIPPED {test}: no checkpoint found ({e:#})");
            None
        }
    }
}

pub fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// max |a - b| / max |b|
pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
