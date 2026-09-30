# Comparison Input Stats

Stats here are only for competitor/legacy benchmark input contracts.  They are
not part of the native SpectralEarth-MM normalization path.

Current staged files:

- `raw_meanstd/ammis`, `raw_meanstd/emit`, `raw_meanstd/gaofen5`: shared
  train-split raw mean/std stats for old SE, DOFA, and Panopticon on these
  unseen sensors
- `hypersigma_scale/ammis`: constant `mu=0`, `sigma=1`
- `hypersigma_scale/emit`: constant `mu=0`, `sigma=32767`
- `hypersigma_scale/gaofen5`: constant `mu=0`, `sigma=1`

Raw mean/std stats for `ammis`, `emit`, and `gaofen5` are shared by old SE,
DOFA, and Panopticon on unseen sensors.  ENMAP Panopticon remains special and
uses `data/panopticon_statistics`.

Put refreshed stats at the registered paths only when they are intended for
real benchmark runs.
