# Failure Detector

A Rust library implementing φ (phi) accrual failure detection for distributed systems. Uses statistical analysis of heartbeat intervals to compute continuous suspicion levels, enabling nuanced failure handling beyond simple binary timeout detectors.

## Why This Matters

Every distributed system needs to know which nodes are alive and which have crashed. The quality of failure detection directly impacts:

- **Availability**: False positives (declaring a healthy node dead) trigger unnecessary failovers
- **Consistency**: False negatives (missing a crashed node) cause stale reads and split brains
- **Performance**: Detection speed determines recovery time

The φ accrual detector, introduced by Hayashibara et al. (2004), is used in production systems including **Apache Cassandra**, **Riak**, **Akka**, and **Orleans**. It provides a continuous suspicion value instead of a binary alive/dead decision, allowing upper-level algorithms to make adaptive decisions.

## Architecture

### HeartbeatWindow

A sliding window of recent heartbeat timestamps. Maintains up to `max_size` samples, discarding the oldest when full. Provides statistical methods:

- `mean_interval()` — Average time between heartbeats
- `std_dev()` — Standard deviation of inter-arrival times
- `intervals()` — Raw inter-arrival time series

### PhiAccrualDetector

The core detector. For each monitored node, it maintains a `HeartbeatWindow` and computes:

```
φ = -log₁₀(P(X ≤ elapsed))
```

where `X` is the random variable representing the next inter-arrival time, estimated from the empirical distribution of past intervals. The probability is approximated using the normal CDF:

```
P(X ≤ t) ≈ Φ((t - μ) / σ)
```

where μ is the mean interval and σ is the standard deviation.

**Interpretation:**
- φ < 1: "This is normal" (expected waiting time)
- φ = 2: "We'd wait this long only 1% of the time"
- φ = 8: "We'd wait this long only 1 in 10⁸ times — probably crashed"

### SuspicionLevel

Classifies the continuous φ value into discrete levels:
- **Healthy** (φ < 1.0): Node is responding normally
- **Suspicious** (1.0 ≤ φ < 4.0): Slight concern, monitor closely
- **LikelyFailed** (4.0 ≤ φ < 8.0): Strong evidence of failure
- **Dead** (φ ≥ 8.0): Node is considered crashed

### AdaptiveDetector

Wraps the phi accrual detector with adaptive window sizing:
- When heartbeats are regular (low variance), a smaller window provides faster detection
- When heartbeats are irregular (high variance), a larger window avoids false positives
- Adaptation is based on the coefficient of variation (CV = σ/μ)

## Usage

```rust
use failure_detector::{PhiAccrualDetector, HeartbeatWindow, SuspicionLevel, AdaptiveDetector};

// Basic phi accrual detection
let mut detector = PhiAccrualDetector::new("monitor", 100, 8.0);
detector.monitor("worker-1");

// Record heartbeats (e.g., from network messages)
for t in (100..1000).step_by(100) {
    detector.heartbeat("worker-1", t as f64);
}

// Check suspicion at various times
let phi_ok = detector.phi("worker-1", 1050.0);
let phi_suspicious = detector.phi("worker-1", 1500.0);
let phi_dead = detector.phi("worker-1", 5000.0);

println!("Normal: φ = {:.2} ({:?})", phi_ok, detector.suspicion("worker-1", 1050.0));
println!("Delayed: φ = {:.2} ({:?})", phi_suspicious, detector.suspicion("worker-1", 1500.0));
println!("Failed: φ = {:.2} ({:?})", phi_dead, detector.suspicion("worker-1", 5000.0));

// Adaptive detector
let mut adaptive = AdaptiveDetector::new("monitor", 5, 50, 8.0);
adaptive.monitor("worker-2");
for t in (100..1000).step_by(100) {
    adaptive.heartbeat("worker-2", t as f64);
}
```

## Mathematical Background

### The Accrual Model

Traditional failure detectors output a binary decision: **trusted** or **suspected**. The accrual model instead outputs a continuous "suspicion level" φ ∈ [0, ∞):

```
φ(t) = -log₁₀(1 - F(t - t_last))
```

where F is the estimated cumulative distribution function of inter-arrival times, and t_last is the time of the last heartbeat.

### Normal Distribution Approximation

We model inter-arrival times as normally distributed: X ~ N(μ, σ²). The CDF is:

```
F(x) = Φ((x - μ) / σ) = (1/2) * [1 + erf((x - μ) / (σ√2))]
```

In practice, we use a logistic approximation:

```
Φ(y) ≈ 1 / (1 + exp(-1.6 * y))
```

This provides a computationally efficient estimate with < 1% error for |y| < 3.

### Phi Interpretation

The φ value has a precise statistical meaning:

```
φ = k ⟹ P(false positive) ≈ 10^(-k)
```

| φ Value | False Positive Rate | Interpretation |
|---------|-------------------|----------------|
| 1 | 10% | Likely normal variation |
| 2 | 1% | Slightly unusual |
| 3 | 0.1% | Suspicious |
| 4 | 0.01% | Very suspicious |
| 8 | 0.000001% | Almost certainly failed |

### Minimum Standard Deviation

When heartbeats arrive at perfectly regular intervals, σ ≈ 0, which would make φ → ∞ at the slightest delay. We clamp σ to a minimum value (default 0.1) to avoid this degenerate case.

### Window Size Trade-off

| Window Size | Detection Speed | False Positive Rate | Accuracy |
|-------------|----------------|---------------------|----------|
| Small (5-10) | Fast | Higher | Lower |
| Medium (50-100) | Moderate | Moderate | Moderate |
| Large (200-500) | Slow | Lower | Higher |

The adaptive detector adjusts window size based on the coefficient of variation:

```
CV = σ / μ
```

- CV < threshold → use smaller window (regular heartbeats)
- CV > threshold → use larger window (irregular heartbeats)

## Comparison with Alternative Approaches

| Approach | Output | Accuracy | Adaptability |
|----------|--------|----------|-------------|
| Simple timeout | Binary | Low | None |
| Exponential moving average | Binary | Medium | Low |
| φ accrual | Continuous | High | Moderate |
| Adaptive φ accrual | Continuous | Very high | High |

## Performance Characteristics

| Operation | Time Complexity | Space per Node |
|-----------|----------------|----------------|
| Record heartbeat | O(1) amortized | O(w) for window size w |
| Compute phi | O(1) | O(w) |
| Mean/StdDev | O(w) | O(1) |

## License

MIT
