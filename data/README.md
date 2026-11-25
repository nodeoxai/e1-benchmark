# E1-A measurement extracts

Thirteen JSON Lines files contain the selected measurements used by the
calculation. They are field-selected extracts of recorded experiments, not
complete event logs. Selected numerical values and their order are unchanged.
Fields and events unrelated to the retained calculations are omitted.

| Folder | Setup | Files |
|---|---|---:|
| same-datacenter | 1, 8 and 16 A40s, coordinator in the GPUs' datacenter, 16–17 October 2026 | 3 |
| coordinator-germany | A40s in one datacenter, coordinator in Germany over the internet; two nights, 13–14 and 16–17 October 2026 | 7 |
| mixed-vendor | Two controls of 3 × A40; one A40 + RTX 4090 + MI300X run across three sites, 14 October 2026 | 3 |

## Retained schema

| Event | Retained fields | Purpose |
|---|---|---|
| run_start | config.num_rounds | Check that all configured rounds are represented |
| round_end | round, seconds | Recalculate round-time medians and speed-up |
| evaluation | status, samples, metrics.accuracy, metrics.samples | Select the final complete evaluation on 3,669 images |

event names each row's type. seconds is a recorded duration in seconds;
metrics.accuracy is a fraction, converted to percent by the script.
MANIFEST.csv lists hashes of these extracts, dates and experiment conditions.
It describes a historical experimental setup, not current production capacity.

## Workload and limits

All runs fine-tuned Google's pretrained ViT-B/16 on Oxford-IIIT Pet with seed
42. Scaling: batch 128, 740 configured total steps per round, round 1 warm-up.
Multi-GPU times use the median of rounds 2–4; the same-site 1-GPU baseline has
only one measured round. Accuracy: batch 32, eight rounds, 240 configured
total steps per round, final top-1 evaluation on 3,669 images. These extracts
are sufficient for recalculating the recorded results, not a complete recipe
for independently rerunning the training implementation.

Night 2 in Germany has no new 1-GPU run; it uses night 1's baseline. One
night-1 input started from an unchanged saved copy of the pretrained model.
The mixed run and the homogeneous scaling runs answer different questions.

## Sources and terms

- Model: Google's ViT-B/16 pretrained on ImageNet-21k, with a 37-class head;
  [google-research/vision_transformer](https://github.com/google-research/vision_transformer), Apache 2.0.
- Dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/),
  3,680 trainval and 3,669 test images, resized to 224 × 224. The source files
  and their third-party terms are separate from these measurement extracts.
- Measurement extracts: [CC BY 4.0](LICENSE), attributed as "Nodeox E1 benchmark
  records, nodeox.com". Code: MIT. Model and dataset files are linked, not included.
