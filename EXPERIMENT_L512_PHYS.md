# Band-only l512 physics-loss snapshot

Snapshot date: 2026-07-02

## Main run

- AE: out/ae_band_l512_phys
- Structure-to-band latent model: out/s2_band_l512_phys
- Test split size: 2591 materials
- Target: 128 x 16 band-only tensor, relative to Fermi level
- Latent dimension: 512
- Added auxiliary losses: Fermi-window, slope, curvature, small gap auxiliary loss

## Test metrics

AE reconstruction upper bound:

- band_mse: 0.110531 eV^2
- band_rmse: 0.241174 eV
- band_mae: 0.191654 eV

Structure-to-band prediction:

- sample mean band_mae: 0.434032 eV
- sample median band_mae: 0.302370 eV
- p90 band_mae: 0.908359 eV
- p95 band_mae: 1.162702 eV
- p99 band_mae: 2.049893 eV
- max band_mae: 3.830445 eV
- global fitting R2: 0.932554
- global fitting RMSE: 0.760217 eV
- global fitting MAE: 0.424834 eV
- global fitting slope: 0.958959
- global fitting intercept: 0.026217 eV

## Notes

Compared with the previous band-only second-stage run, the l512 physics-loss run improves mean band MAE from about 0.458 eV to about 0.434 eV, and improves the AE reconstruction upper bound from about 0.205 eV to about 0.192 eV. The remaining main limitation is a long-tailed error distribution and regression-to-the-mean behavior at large absolute energies.
