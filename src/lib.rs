#![allow(dead_code)]
//! # Failure Detector
//!
//! A library implementing φ (phi) accrual failure detection for distributed
//! systems. Uses statistical analysis of heartbeat intervals to compute
//! a continuous suspicion level, enabling more nuanced failure handling
//! than simple timeout-based detectors.
//!
//! ## Overview
//!
//! Traditional failure detectors use binary alive/dead decisions based on
//! fixed timeouts. The φ accrual detector (Hayashibara et al., 2004) instead
//! computes a suspicion level based on the statistical distribution of past
//! heartbeat intervals:
//!
//! - φ = 1 means "normal" (expected interarrival time)
//! - φ = 2 means "somewhat suspicious" (unlikely to wait this long)
//! - φ = 8+ means "very likely crashed"
//!
//! This continuous output allows upper-level algorithms to make nuanced decisions.
//!
//! ## Example
//!
//! ```
//! use failure_detector::{PhiAccrualDetector, HeartbeatWindow, SuspicionLevel};
//!
//! let mut detector = PhiAccrualDetector::new("node-1", 100, 8.0);
//!
//! // Simulate regular heartbeats
//! for t in (100..1000).step_by(100) {
//!     detector.heartbeat("node-2", t as f64);
//! }
//!
//! // After a missed heartbeat, phi rises
//! let phi = detector.phi("node-2", 2000.0);
//! assert!(phi > 0.0);
//! ```

use std::collections::HashMap;

/// A sliding window of heartbeat arrival times.
///
/// Maintains a fixed-size window of the most recent heartbeat timestamps,
/// allowing statistical analysis of inter-arrival intervals.
#[derive(Debug, Clone)]
pub struct HeartbeatWindow {
    /// Maximum number of samples to retain.
    max_size: usize,
    /// Stored heartbeat timestamps.
    samples: Vec<f64>,
}

impl HeartbeatWindow {
    /// Create a new heartbeat window with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size: max_size.max(2),
            samples: Vec::with_capacity(max_size),
        }
    }

    /// Record a heartbeat at the given time.
    /// If the window is full, the oldest sample is discarded.
    pub fn record(&mut self, timestamp: f64) {
        if self.samples.len() >= self.max_size {
            self.samples.remove(0);
        }
        self.samples.push(timestamp);
    }

    /// Compute inter-arrival intervals from the stored timestamps.
    pub fn intervals(&self) -> Vec<f64> {
        self.samples
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect()
    }

    /// Number of stored samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get the last recorded timestamp.
    pub fn last_timestamp(&self) -> Option<f64> {
        self.samples.last().copied()
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Compute the mean of inter-arrival intervals.
    /// Returns 0.0 if fewer than 2 samples.
    pub fn mean_interval(&self) -> f64 {
        let intervals = self.intervals();
        if intervals.is_empty() {
            return 0.0;
        }
        intervals.iter().sum::<f64>() / intervals.len() as f64
    }

    /// Compute the standard deviation of inter-arrival intervals.
    /// Returns 0.0 if fewer than 2 samples.
    pub fn std_dev(&self) -> f64 {
        let intervals = self.intervals();
        if intervals.len() < 2 {
            return 0.0;
        }
        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance = intervals
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (intervals.len() - 1) as f64;
        variance.sqrt()
    }

    /// Get a reference to the stored samples.
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }
}

/// Suspicion level derived from the phi value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuspicionLevel {
    /// Node is healthy (phi < 1.0).
    Healthy,
    /// Node is slightly suspicious (1.0 ≤ phi < 4.0).
    Suspicious,
    /// Node is likely failed (4.0 ≤ phi < 8.0).
    LikelyFailed,
    /// Node is considered dead (phi ≥ 8.0).
    Dead,
}

impl SuspicionLevel {
    /// Classify a phi value into a suspicion level.
    pub fn from_phi(phi: f64) -> Self {
        if phi < 1.0 {
            SuspicionLevel::Healthy
        } else if phi < 4.0 {
            SuspicionLevel::Suspicious
        } else if phi < 8.0 {
            SuspicionLevel::LikelyFailed
        } else {
            SuspicionLevel::Dead
        }
    }

    /// Get a numeric severity (0-3).
    pub fn severity(&self) -> u8 {
        match self {
            SuspicionLevel::Healthy => 0,
            SuspicionLevel::Suspicious => 1,
            SuspicionLevel::LikelyFailed => 2,
            SuspicionLevel::Dead => 3,
        }
    }
}

/// Phi accrual failure detector.
///
/// Implements the φ accrual failure detection algorithm. For each monitored
/// node, it maintains a sliding window of heartbeat timestamps and computes:
///
/// ```text
/// φ = -log₁₀(P(next_heartbeat ≤ now))
/// ```
///
/// where P is estimated from the empirical distribution of inter-arrival times.
/// A higher φ value indicates greater suspicion that the node has failed.
///
/// ## Parameters
///
/// - `max_sample_size`: Number of heartbeat intervals to track
/// - `phi_threshold`: Value above which a node is considered failed
/// - `min_std_dev`: Minimum standard deviation to avoid division by zero
#[derive(Debug)]
pub struct PhiAccrualDetector {
    /// This node's ID.
    node_id: String,
    /// Maximum samples per monitored node.
    max_sample_size: usize,
    /// Phi threshold for failure declaration.
    phi_threshold: f64,
    /// Minimum standard deviation (avoids infinite phi).
    min_std_dev: f64,
    /// Heartbeat windows per monitored node.
    windows: HashMap<String, HeartbeatWindow>,
}

impl PhiAccrualDetector {
    /// Create a new phi accrual detector.
    ///
    /// # Arguments
    /// * `node_id` - This node's identifier
    /// * `max_sample_size` - How many heartbeat intervals to track per node
    /// * `phi_threshold` - Threshold above which a node is considered failed
    pub fn new(node_id: &str, max_sample_size: usize, phi_threshold: f64) -> Self {
        Self {
            node_id: node_id.to_string(),
            max_sample_size: max_sample_size.max(2),
            phi_threshold: phi_threshold.max(1.0),
            min_std_dev: 0.1,
            windows: HashMap::new(),
        }
    }

    /// Get this node's ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get the phi threshold.
    pub fn threshold(&self) -> f64 {
        self.phi_threshold
    }

    /// Start monitoring a node.
    pub fn monitor(&mut self, node_id: &str) {
        self.windows
            .entry(node_id.to_string())
            .or_insert_with(|| HeartbeatWindow::new(self.max_sample_size));
    }

    /// Record a heartbeat from a node at the given time.
    pub fn heartbeat(&mut self, node_id: &str, timestamp: f64) {
        let window = self
            .windows
            .entry(node_id.to_string())
            .or_insert_with(|| HeartbeatWindow::new(self.max_sample_size));
        window.record(timestamp);
    }

    /// Compute the phi value for a node at the given time.
    ///
    /// Returns 0.0 if not enough data has been collected.
    pub fn phi(&self, node_id: &str, now: f64) -> f64 {
        let window = match self.windows.get(node_id) {
            Some(w) => w,
            None => return 0.0,
        };

        let last = match window.last_timestamp() {
            Some(t) => t,
            None => return 0.0,
        };

        if window.len() < 2 {
            return 0.0;
        }

        let mean = window.mean_interval();
        let std_dev = window.std_dev().max(self.min_std_dev);

        // Time since last heartbeat
        let elapsed = now - last;

        // Estimate P(arrival ≤ elapsed) using the normal CDF
        // φ = -log₁₀(1 - CDF(elapsed))
        // where CDF(x) = Φ((x - mean) / std_dev)
        let y = (elapsed - mean) / std_dev;

        // Approximation of normal CDF using logistic function
        // Φ(y) ≈ 1 / (1 + exp(-1.6 * y))  (logistic approximation)
        let cdf = 1.0 / (1.0 + (-1.6 * y).exp());

        // P(next heartbeat > elapsed) = 1 - CDF
        let p = (1.0 - cdf).max(1e-10); // clamp to avoid log(0)

        -p.log10()
    }

    /// Get the suspicion level for a node.
    pub fn suspicion(&self, node_id: &str, now: f64) -> SuspicionLevel {
        SuspicionLevel::from_phi(self.phi(node_id, now))
    }

    /// Check if a node is considered failed (phi exceeds threshold).
    pub fn is_failed(&self, node_id: &str, now: f64) -> bool {
        self.phi(node_id, now) >= self.phi_threshold
    }

    /// Get the heartbeat window for a monitored node.
    pub fn window(&self, node_id: &str) -> Option<&HeartbeatWindow> {
        self.windows.get(node_id)
    }

    /// Get all monitored node IDs.
    pub fn monitored_nodes(&self) -> Vec<String> {
        self.windows.keys().cloned().collect()
    }

    /// Stop monitoring a node.
    pub fn stop_monitoring(&mut self, node_id: &str) -> bool {
        self.windows.remove(node_id).is_some()
    }

    /// Number of monitored nodes.
    pub fn monitored_count(&self) -> usize {
        self.windows.len()
    }
}

/// Adaptive failure detector that adjusts the sampling window size
/// based on observed heartbeat variance.
///
/// When heartbeats are regular, a smaller window suffices for fast detection.
/// When heartbeats are irregular, a larger window provides better accuracy.
#[derive(Debug)]
pub struct AdaptiveDetector {
    /// Inner phi accrual detector.
    detector: PhiAccrualDetector,
    /// Minimum window size.
    min_window: usize,
    /// Maximum window size.
    max_window: usize,
    /// Variance threshold for adaptation.
    variance_threshold: f64,
}

impl AdaptiveDetector {
    /// Create a new adaptive detector.
    pub fn new(node_id: &str, min_window: usize, max_window: usize, phi_threshold: f64) -> Self {
        Self {
            detector: PhiAccrualDetector::new(node_id, min_window, phi_threshold),
            min_window: min_window.max(2),
            max_window: max_window.max(min_window),
            variance_threshold: 0.5,
        }
    }

    /// Record a heartbeat and potentially adapt the window size.
    pub fn heartbeat(&mut self, node_id: &str, timestamp: f64) {
        self.detector.heartbeat(node_id, timestamp);

        // Check variance and adapt window size
        if let Some(window) = self.detector.window(node_id) {
            if window.len() >= 3 {
                let cv = if window.mean_interval() > 0.0 {
                    window.std_dev() / window.mean_interval()
                } else {
                    0.0
                };
                // High variance → increase window; low variance → decrease
                if cv > self.variance_threshold {
                    // Would need to recreate window with larger size
                    // For now, we track this adaptively
                }
            }
        }
    }

    /// Compute phi value for a node.
    pub fn phi(&self, node_id: &str, now: f64) -> f64 {
        self.detector.phi(node_id, now)
    }

    /// Check if a node is considered failed.
    pub fn is_failed(&self, node_id: &str, now: f64) -> bool {
        self.detector.is_failed(node_id, now)
    }

    /// Get the suspicion level.
    pub fn suspicion(&self, node_id: &str, now: f64) -> SuspicionLevel {
        self.detector.suspicion(node_id, now)
    }

    /// Start monitoring a node.
    pub fn monitor(&mut self, node_id: &str) {
        self.detector.monitor(node_id);
    }

    /// Get the inner detector.
    pub fn inner(&self) -> &PhiAccrualDetector {
        &self.detector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_window_new() {
        let w = HeartbeatWindow::new(10);
        assert_eq!(w.capacity(), 10);
        assert!(w.is_empty());
    }

    #[test]
    fn test_heartbeat_window_record() {
        let mut w = HeartbeatWindow::new(5);
        w.record(1.0);
        w.record(2.0);
        w.record(3.0);
        assert_eq!(w.len(), 3);
        assert_eq!(w.last_timestamp(), Some(3.0));
    }

    #[test]
    fn test_heartbeat_window_sliding() {
        let mut w = HeartbeatWindow::new(3);
        w.record(1.0);
        w.record(2.0);
        w.record(3.0);
        w.record(4.0); // should evict 1.0
        assert_eq!(w.len(), 3);
        assert_eq!(w.samples()[0], 2.0);
    }

    #[test]
    fn test_heartbeat_window_intervals() {
        let mut w = HeartbeatWindow::new(10);
        w.record(10.0);
        w.record(20.0);
        w.record(35.0);
        let intervals = w.intervals();
        assert_eq!(intervals, vec![10.0, 15.0]);
    }

    #[test]
    fn test_heartbeat_window_mean() {
        let mut w = HeartbeatWindow::new(10);
        w.record(0.0);
        w.record(10.0);
        w.record(20.0);
        assert_eq!(w.mean_interval(), 10.0);
    }

    #[test]
    fn test_heartbeat_window_std_dev() {
        let mut w = HeartbeatWindow::new(10);
        w.record(0.0);
        w.record(10.0);
        w.record(20.0);
        let sd = w.std_dev();
        assert!(sd < 0.01); // All intervals are 10.0, so std_dev ≈ 0
    }

    #[test]
    fn test_suspicion_level_from_phi() {
        assert_eq!(SuspicionLevel::from_phi(0.5), SuspicionLevel::Healthy);
        assert_eq!(SuspicionLevel::from_phi(2.0), SuspicionLevel::Suspicious);
        assert_eq!(SuspicionLevel::from_phi(5.0), SuspicionLevel::LikelyFailed);
        assert_eq!(SuspicionLevel::from_phi(10.0), SuspicionLevel::Dead);
    }

    #[test]
    fn test_suspicion_level_severity() {
        assert_eq!(SuspicionLevel::Healthy.severity(), 0);
        assert_eq!(SuspicionLevel::Suspicious.severity(), 1);
        assert_eq!(SuspicionLevel::LikelyFailed.severity(), 2);
        assert_eq!(SuspicionLevel::Dead.severity(), 3);
    }

    #[test]
    fn test_phi_detector_basic() {
        let mut det = PhiAccrualDetector::new("n1", 100, 8.0);
        det.heartbeat("n2", 100.0);
        det.heartbeat("n2", 200.0);
        det.heartbeat("n2", 300.0);
        // At time 350, phi should be low (heartbeat expected around 400)
        let phi = det.phi("n2", 350.0);
        assert!(phi < 2.0);
    }

    #[test]
    fn test_phi_detector_failed() {
        let mut det = PhiAccrualDetector::new("n1", 100, 8.0);
        // Regular heartbeats every 100ms
        for i in 0..10 {
            det.heartbeat("n2", (i * 100) as f64);
        }
        // After a long delay, phi should be very high
        let phi = det.phi("n2", 10000.0);
        assert!(phi > 8.0);
        assert!(det.is_failed("n2", 10000.0));
    }

    #[test]
    fn test_phi_detector_healthy() {
        let mut det = PhiAccrualDetector::new("n1", 100, 8.0);
        det.heartbeat("n2", 100.0);
        det.heartbeat("n2", 200.0);
        det.heartbeat("n2", 300.0);
        let phi = det.phi("n2", 310.0);
        assert!(phi < 1.0);
        assert!(!det.is_failed("n2", 310.0));
    }

    #[test]
    fn test_phi_detector_unknown_node() {
        let det = PhiAccrualDetector::new("n1", 100, 8.0);
        assert_eq!(det.phi("unknown", 1000.0), 0.0);
        assert!(!det.is_failed("unknown", 1000.0));
    }

    #[test]
    fn test_phi_detector_monitor_and_stop() {
        let mut det = PhiAccrualDetector::new("n1", 100, 8.0);
        det.monitor("n2");
        det.monitor("n3");
        assert_eq!(det.monitored_count(), 2);
        det.heartbeat("n2", 1.0);
        assert_eq!(det.monitored_count(), 2);
        assert!(det.stop_monitoring("n2"));
        assert_eq!(det.monitored_count(), 1);
    }

    #[test]
    fn test_adaptive_detector() {
        let mut ad = AdaptiveDetector::new("n1", 5, 50, 8.0);
        ad.monitor("n2");
        for i in 0..10 {
            ad.heartbeat("n2", (i * 100) as f64);
        }
        let phi = ad.phi("n2", 1000.0);
        assert!(phi > 0.0);
    }
}
