use ahash::RandomState;
use once_cell::sync::Lazy;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hash::Hash;

pub static FIXED_RANDOM_STATE: Lazy<RandomState> = Lazy::new(|| {
    let mut rng = StdRng::seed_from_u64(0);
    RandomState::with_seeds(rng.random(), rng.random(), rng.random(), rng.random())
});

pub fn compute_hash<T: Hash + ?Sized>(value: &T) -> u64 {
    FIXED_RANDOM_STATE.hash_one(value)
}
