# Algorithm

## Parameterization

For width H, there are 59H weights, H hidden gains, and H+10 biases: 61H+10 discrete variables (3,914 at H=64). Codes are stored as INT8. Weights are initially uniform ternary; gains start at 1 and biases at 0.

The hidden gain is denoted λ in the README and architecture diagram (`g` in the implementation). Regenerate the PNG/SVG diagram with `python scripts/draw_architecture.py` after installing the plots extra.

Calibration uses all training images. δ is the RMS of a layer's inputs, shared across its neurons. τ is each neuron's RMS initial preactivation, including its bias. An exactly zero τ is replaced by 1. These constants never change. Output logits are `(Vh + δ_out b_out)/τ_out`; there is no output activation or explicit softmax.

## Candidate selection and scoring

1. Choose K from powers of two up to the power nearest 5% of the weight count (ties favor the lower power). The initial nominal K is 4; width 64 gives a maximum of 128.
2. Select exactly K distinct weight positions without replacement using a weighted Fenwick tree. Biases and gains have separate confidence-weighted Bernoulli selection budgets; K does not count these auxiliary variables.
3. Draw 24 logical candidates uniformly from the three codes at the shared selected positions. Draws may coincide with the current network or one another. Evaluate only distinct new networks, retaining logical multiplicities for confidence calculations.
4. Score with negative mean cross-entropy on the same batch. When multiple distinct candidates exist, tiled incremental scoring reuses the current network's intermediate calculations. It does not retain all training activations between rounds.
5. If K>1 and at least two distinct new candidates exist, form a strict consensus of candidates that improve the current score. Retain a proposed value only when every improving candidate agrees. Evaluate this network if its score is not already known.
6. Compare the current network, candidates and consensus. Accept only a strict improvement. A round uses at most 26 physical network evaluations: current, 24 candidates, and consensus.

The native scorer dispatches between AVX2 and AVX512 according to CPU support. Its tiled scratch buffers bound activation workspace. This implementation is specialized to the single-hidden-layer model; it is not a general large-model or GPU trainer.

## Confidence

For each selected variable, compare candidate scores retaining its current value with scores changing it, using the reference's graduated rank signal. Let a be the number of the 24 candidates retaining the current value. Store scaled counters A=24E and B=24M:

- Add the signed graduated rank increment to A.
- Add `a(24−a)` to B.
- Reset both when the variable's retained value changes.

The selection preference is

$$q_i=0.001+0.999\exp\left[-\operatorname{clip}\left(\frac{A_i}{\sqrt{24(B_i+72)}},-3,3\right)\right].$$

Preferences are normalized through weighted sampling; 0.001 is a floor on the preference, not a fixed absolute inclusion probability. This confidence is a heuristic, not a calibrated posterior probability.

Both counters are INT32 without forgetting. Before an update exceeds their range, the pair is promoted to a sparse INT64 extension. Reset removes the extension. INT64 overflow is rejected rather than wrapped. Snapshots include extensions. Nominal counter storage at width 64 is 31,312 bytes; that excludes weights, the sampling tree, data and scorer workspace.

## K adaptation

A window contains the last 512 global rounds. Each K records total positive improvement and deterministic predicted evaluation cost; failures add cost and zero gain. Exploitation selects the highest gain/cost estimate. The policy uses the nominal arm 90% of the time and samples uniformly among the other arms 10% of the time.

The retained controller updates K at round 64 and every 128 rounds thereafter. This cadence comes from alternating controller slots; population adaptation is disabled and the actual population is fixed at 24. Its legacy population label is not an adaptive setting.

`sts/cost_model.json` contains frozen CPU cost coefficients. Runtime measurements never alter this controller. This improves fixed-round reproducibility, but the coefficients need not predict cost accurately on every CPU.

## Training batches

Each round uses a renewed uniform subset of the 59,000 training images, starting with 32. An independently shuffled nested training probe has the same size as the current batch. Every 16 rounds, compare successive windows of four probe losses. Double the batch when the newer mean exceeds the previous mean by more than 1%. Reset probe history after growth. The cap is the full training set, at which point probing stops. These decisions use training images, never validation or test data.

Improvement on one renewed batch does not imply monotonically decreasing full-training or validation loss. Fixed-round reproduction requires matching the model, seed, native arithmetic, hardware and thread count. A time-limited run additionally depends on system load.
