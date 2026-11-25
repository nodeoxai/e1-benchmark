# Recalculate the E1-A measurements

Python 3.9 or later, standard library only. Clone the complete repository, then:

```bash
git clone https://github.com/nodeoxai/e1-benchmark.git
cd e1-benchmark
python3 recalculate.py --check expected/results.json
```

This recalculates recorded measurements; it does not rerun training. After
cloning, no network connection, GPU or Nodeox account is needed.

## Expected output

```text
E1 recalculation: ViT-B/16 on Oxford-IIIT Pet, preliminary

Scaling, coordinator in the GPUs' datacenter (16-17 Oct 2026)
  GPUs   round, s   speed-up   efficiency
     1    893.143      1.00x      100.0%
     8    128.266      6.96x       87.0%
    16     79.732     11.20x       70.0%
  The 1-GPU time is one measured round; the others are the median of rounds 2-4.

Scaling, coordinator in Germany, night 1 (13-14 Oct 2026)
  GPUs   round, s   speed-up   efficiency
     1    906.670      1.00x      100.0%
     2    463.097      1.96x       97.9%
     4    233.057      3.89x       97.3%
     8    130.536      6.95x       86.8%
    16     84.169     10.77x       67.3%

Scaling, coordinator in Germany, night 2 (16-17 Oct 2026), against night 1's 1-GPU time
  GPUs   round, s   speed-up   efficiency
     8    131.602      6.89x       86.1%
    16     94.066      9.64x       60.2%

16 GPUs in the GPUs' datacenter against night 1's 1-GPU time: 11.37x, 71.1%

Mixed-vendor accuracy, final top-1 on 3669 test images (14 Oct 2026)
  NVIDIA-only control A   93.4042%
  NVIDIA-only control B   93.5132%
  A40 + RTX 4090 + MI300X 93.0771%
  Gap from the controls' mean 0.3816 percentage points; tolerance 0.5000 pp; within: True; both controls at least 90%: True

Check passed: every value matches expected/results.json.
```

## Inputs and methods

| Result | Files under data/ | Calculation |
|---|---|---|
| Same-site scaling | same-datacenter/a40x01, a40x08, a40x16 | 1-GPU round 2 / median of multi-GPU rounds 2–4; divide by GPU count for efficiency |
| Germany, night 1 | coordinator-germany/night1-a40x01/02/04/08/16 | Median of rounds 2–4; night 1's 1-GPU time is the baseline |
| Germany, night 2 | coordinator-germany/night2-a40x08/16 | Median of rounds 2–4 against night 1's 1-GPU baseline |
| Alternative baseline | coordinator-germany/night1-a40x01 and same-datacenter/a40x16 | Night 1's 1-GPU time / same-site 16-GPU time |
| Mixed-vendor accuracy | mixed-vendor/control-a, control-b, mixed | Final complete evaluation on all 3,669 test images; accuracy × 100; gap from controls' mean |
| Predefined accuracy test | The same three accuracy extracts | Gap ≤ max(0.5 percentage points, control difference); both controls ≥ 90% |

Filenames have the .jsonl suffix. A missing or duplicated round, absent input
or incomplete final evaluation fails visibly. --json FILE also saves results.
The one-round same-site 1-GPU baseline is an explicit exception. These are
measurement extracts, not full event logs; per-phase exchange timings are
outside this release. See [data/README.md](data/README.md) and
[MANIFEST.csv](MANIFEST.csv) for the schema and file hashes.
