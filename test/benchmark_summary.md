> **Stale — regenerate before quoting.** These timings predate the reduction
> changes that cut collectives per call (`mean` 5 → 2, `sum` 3 → 2, `min` 3 → 2,
> `std` 10 → 4) and the halo-based `interp`. Rerun `test/test.sh` on the target
> system; timings from an oversubscribed single-core machine are not comparable.

Combined from 1 rank-count run(s): [8]. 11 (method, ranks) rows, 1 slower-than-native at their run's problem size.

| Method | Ranks | Native Xarray | MPI Xarray | Speedup | Accuracy | Dtype |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `coarsen_mean` | 8 | 0.1920 s | 0.0393 s | 4.89x | PASS | PASS |
| `diff` | 8 | 0.0518 s | 0.0195 s | 2.66x | PASS | PASS |
| `differentiate` | 8 | 0.2721 s | 0.0476 s | 5.72x | PASS | PASS |
| `isel` | 8 | 0.0001 s | 0.0003 s | 0.37x SLOWER | PASS | PASS |
| `mean` | 8 | 0.0313 s | 0.0270 s | 1.16x | PASS | PASS |
| `distribute_data` | 8 | n/a | 0.0791 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `np.log` | 8 | 0.1601 s | 0.0205 s | 7.80x | PASS | PASS |
| `np.multiply` | 8 | 0.0460 s | 0.0057 s | 8.11x | PASS | PASS |
| `np.sqrt` | 8 | 0.0987 s | 0.0160 s | 6.16x | PASS | PASS |
| `rolling_mean` | 8 | 0.1235 s | 0.0302 s | 4.08x | PASS | PASS |
| `sum` | 8 | 0.1370 s | 0.0202 s | 6.78x | PASS | PASS |
