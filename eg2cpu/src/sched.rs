//! Process-wide scheduling of forward passes requested from outside the rayon pool.
//!
//! A lone request runs at once, on a pool sized to its length. Requests that arrive while another is running
//! queue up instead, per caller. Whichever waiting caller finds nothing running becomes the dispatcher: it
//! waits a short collection window so stragglers can join (only when fewer than `max_in_flight()` are queued),
//! then runs `max_in_flight()` workers on the global pool. Each worker repeatedly takes the next job round-robin
//! across callers, runs it, and publishes its result at once, until the queue is empty. Only one solo pass or one
//! dispatch runs at a time, so forward passes in flight never exceed `max_in_flight()` across the process. A
//! request that arrives mid-dispatch takes the next free worker slot, so a large batch from one caller delays
//! others' single requests by about one pass, not by the whole batch.

use crate::model::{Model, Output, max_in_flight};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

pub(crate) struct Scheduler {
    state: Mutex<State>,
    cv: Condvar,
}

#[derive(Default)]
struct State {
    /// Queued jobs per caller, each caller's in submission order.
    queue: BTreeMap<u64, VecDeque<Job>>,
    queued: usize,
    /// The caller served last, for round-robin.
    last_caller: u64,
    /// A solo pass or a dispatch is running.
    busy: bool,
    next_job: u64,
    next_caller: u64,
    /// Finished jobs waiting for their caller; None if the pass panicked.
    done: HashMap<u64, Option<Output>>,
}

struct Job {
    id: u64,
    ids: Vec<u32>,
}

impl State {
    /// The next job, taking callers in turn (the first caller after the one served last).
    fn pick(&mut self) -> Option<Job> {
        let caller = *self.queue.range(self.last_caller + 1..).next().or_else(|| self.queue.iter().next())?.0;
        let jobs = self.queue.get_mut(&caller)?;
        let job = jobs.pop_front();
        if jobs.is_empty() {
            self.queue.remove(&caller);
        }
        self.last_caller = caller;
        self.queued -= 1;
        job
    }
}

/// `$EG2_LINGER_US`, default 2000: how long an idle dispatch worker waits for another request before leaving.
fn linger() -> Duration {
    let us = std::env::var("EG2_LINGER_US").ok().and_then(|v| v.parse().ok()).unwrap_or(2000u64);
    Duration::from_micros(us)
}

/// `$EG2_BATCH_WINDOW_US`, default 1000: how long a dispatcher waits for more requests before starting when
/// fewer than `max_in_flight()` are queued. Only paid under contention; a lone request never waits.
fn window() -> Duration {
    let us = std::env::var("EG2_BATCH_WINDOW_US").ok().and_then(|v| v.parse().ok()).unwrap_or(1000u64);
    Duration::from_micros(us)
}

impl Scheduler {
    pub(crate) fn new() -> Self {
        Self { state: Mutex::new(State::default()), cv: Condvar::new() }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Encode each token-id sequence; outputs come back in order.
    pub(crate) fn run(&self, model: &Model, texts: &[&[u32]]) -> Vec<Output> {
        let mut st = self.lock();
        if texts.len() == 1 && !st.busy && st.queued == 0 {
            // Uncontended: run now, on the pool sized for this text. Arrivals meanwhile queue up.
            st.busy = true;
            drop(st);
            let out = catch_unwind(AssertUnwindSafe(|| model.forward_solo(texts[0])));
            self.lock().busy = false;
            self.cv.notify_all();
            return vec![out.unwrap_or_else(|e| resume_unwind(e))];
        }

        let caller = st.next_caller;
        st.next_caller += 1;
        let mine: Vec<u64> = (0..texts.len() as u64).map(|i| st.next_job + i).collect();
        st.next_job += texts.len() as u64;
        // A caller's own jobs go longest first, so its batch doesn't end waiting on one long text.
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(texts[i].len()));
        st.queue.insert(caller, order.iter().map(|&i| Job { id: mine[i], ids: texts[i].to_vec() }).collect());
        st.queued += texts.len();
        self.cv.notify_all();

        loop {
            if mine.iter().all(|i| st.done.contains_key(i)) {
                let outs: Vec<Option<Output>> = mine.iter().map(|i| st.done.remove(i).unwrap()).collect();
                drop(st);
                return outs.into_iter().map(|o| o.expect("a forward pass for this request panicked")).collect();
            }
            if !st.busy && st.queued > 0 {
                st.busy = true;
                let cap = max_in_flight();
                let w = window();
                let wait = st.queued < cap && !w.is_zero();
                drop(st);
                if wait {
                    std::thread::sleep(w);
                }
                self.dispatch(model, cap);
                st = self.lock();
                st.busy = false;
                self.cv.notify_all();
                continue;
            }
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// `cap` workers on the global pool take jobs round-robin until the queue is empty, publishing each result.
    /// A worker that finds the queue empty waits up to `linger()` for another request before leaving: callers that
    /// resubmit as soon as their result arrives (many threads each sending one text at a time) otherwise find the
    /// workers gone and the whole round restarts from scratch.
    fn dispatch(&self, model: &Model, cap: usize) {
        let linger = linger();
        rayon::scope(|s| {
            for _ in 0..cap.max(1) {
                s.spawn(|_| {
                    loop {
                        // Its own block, so the lock is released before the pass runs (a guard in a
                        // `while let` condition would live for the whole loop body).
                        let next = {
                            let mut st = self.lock();
                            let mut job = st.pick();
                            let until = std::time::Instant::now() + linger;
                            while job.is_none() && !linger.is_zero() {
                                let left = until.saturating_duration_since(std::time::Instant::now());
                                if left.is_zero() {
                                    break;
                                }
                                st = self.cv.wait_timeout(st, left).unwrap_or_else(|e| e.into_inner()).0;
                                job = st.pick();
                            }
                            job
                        };
                        let Some(job) = next else { break };
                        let out = catch_unwind(AssertUnwindSafe(|| model.forward_in_pool(&job.ids))).ok();
                        self.lock().done.insert(job.id, out);
                        self.cv.notify_all();
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_takes_callers_in_turn() {
        let mut st = State::default();
        let jobs = |ids: &[u64]| ids.iter().map(|&id| Job { id, ids: vec![] }).collect::<VecDeque<_>>();
        st.queue.insert(1, jobs(&[10, 11, 12, 13]));
        st.queue.insert(2, jobs(&[20]));
        st.queue.insert(3, jobs(&[30, 31]));
        st.queued = 7;
        let order: Vec<u64> = std::iter::from_fn(|| st.pick()).map(|j| j.id).collect();
        assert_eq!(order, vec![10, 20, 30, 11, 31, 12, 13]);
        assert_eq!(st.queued, 0);
        assert!(st.queue.is_empty());
    }
}
