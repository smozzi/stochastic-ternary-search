# Benchmark protocols

## Corrected comparison

The introductory benchmark now uses the current STS checkpoints and independently confirmed Adam/STE models. See [the stability review](ste_review.md) for the full tuning study, corrected protocols, both STE families, audits and reproduction commands. The following historical sections are preserved as archive context.

## Historical V2 / Adam comparison

Architecture 49→64→10; MNIST 7×7 from 4×4 average pooling; normalization (x−0.1307)/0.3081. The same deterministic 59,000/1,000/10,000 training/validation/test split is used. Intel i7-11700, eight threads, 240 seconds per run, seeds 500–502. Historical timers include fresh model/optimizer construction (and STS calibration); dataset loading and disposable warmup are excluded. Checkpoint evaluation overhead is included in the experiment drivers' recorded timing. STS V2 measurements were reused from audited historical runs rather than recomputed simultaneously with Adam.

STS has 3,914 discrete variables, calibrated quadratic activations, fixed per-neuron normalization, and discrete biases/gains. Adam models have 3,850 FP32 latent parameters, ReLU activations, floating biases, and no STS calibration or gains. An Adam update includes forward and backward passes; a search round evaluates a population. Their update counts are not interchangeable.

Adam uses batch 128 and learning rate 0.003. STE uses batches 128 and 5,000 with learning rate 0.001. Both use Adam betas (0.9,0.999), epsilon 1e−8, coupled weight decay 1e−4 and foreach=False. No scheduler, dropout or batch normalization is used.

Rates were selected from {0.001,0.003,0.01} on two pilot seeds (950,951), with 20 seconds per condition/rate/seed: 360 seconds of tuning separate from the reported confirmation budgets. This does not establish optimal rates for 240-second runs.

STE uses a TWN-like per-output-row threshold: retain |w|>0.7·mean(|w|), compute α as the retained mean absolute weight, and forward with α times ternary codes. Codes and scales are detached, with an identity surrogate derivative for latent weights. Biases stay floating point. Quantization begins immediately, without floating-point pretraining. It is a limited baseline, not a full reproduction of every technique in [Ternary Weight Networks](https://arxiv.org/abs/1605.04711).

Terminal metrics describe the model at 240 seconds. Selected metrics choose the best recorded validation-loss checkpoint independently for each run, then evaluate its test accuracy. Test scores never select checkpoints. STE's early optimum and later regression must both be considered; terminal results alone would obscure that behavior.

CSV accuracy values are fractions, not percentages. The historical `ideal_inference_bytes` column estimates packed payloads only: two bits per discrete code plus required FP32 scales/biases. It excludes runtime workspace, serialization overhead and training state; the implementation stores codes as INT8 and does not implement two-bit packing. All reported deviations are sample standard deviations across seeds. Plot bands are ±1 sample SD, not confidence intervals. The additional fixed-batch STS condition remains in the CSV archive but is omitted from the introductory figure.

## Current INT32 comparison

Same architecture and split; 600 seconds per run, seeds 500–504. Current INT32 confidence without forgetting is compared with confidence disabled, keeping search/scoring and adaptive batching otherwise unchanged. Ten terminal runs were independently audited. No counter needed INT64 extension during these runs.

Mean terminal rounds: 512,047 with confidence; 468,523 without. This difference also reflects adaptive batch trajectories, especially one no-confidence seed reaching a larger batch; it does not demonstrate an equivalent native-kernel speed difference.

The paired mean validation-loss improvement is 0.014665, sample SD 0.013267. A two-sided t interval with four degrees of freedom is [−0.001808,0.031137]. Five seeds are insufficient to make a strong general claim.

The current results originate from research commit 5b00172276df1ab961dc50009d504bd9c5e587e8. Source artifact hashes are preserved in results/source_manifest.json; internal filesystem paths were removed from exported tables. The standalone port was separately checked against that reference for 1,024 rounds at all three supported widths. The new CLI has simpler reporting overhead, so its wall-clock throughput need not duplicate archival drivers exactly.

## Reproduction scope

Build and run the current STS implementation using the README instructions. Generate figures from the exported tables with `pip install -e '.[plots]'` and `python scripts/plot_results.py`. The corrected comparators are executable with scripts/benchmark_baselines.py. Historical baseline results remain measured CSV artifacts; their original harness is not included. No claim is made that the current STS beats Adam at a matched 600-second budget.

## Introductory MNIST examples

The montage uses the audited seed-500 STS checkpoint at 240 seconds, with no retraining. Its inference codes and fixed calibration are exported in `results/mnist_example_model.json`; the checkpoint hash links it to `results/sts_240_validation.json`. Predictions were checked using independent NumPy inference against PyTorch on all 10,000 test images, with identical classifications.

Selection seed 20261004 draws 19 examples uniformly without replacement from correct predictions and one from incorrect predictions, then shuffles them. This is a correctness-stratified illustration, not a random sample estimating accuracy. Test indices and labels are recorded in `results/mnist_examples.json`. Pixel intensities use the actual 4×4 average pooling from 28×28 to 7×7, with nearest-neighbor display and no visual smoothing.

Regenerate with `python scripts/plot_mnist_examples.py --data data` after installing the plots extra. The script downloads and checksums the official test files if absent; it does not train or modify the model.
