# DADAnalyse
DADAnalyse is a module to process, analyze, and visualize voltage data from.DADA files. It handles data loading, processing (e.g., cross-power spectra, visibilities), and various diagnostic plots, including antenna power spectra. 

## Reading sections of a file
The file is memory-mapped, so only the requested time range, antennas and channels are read from disk.

```bash
python DADAnalyse.py file.dada --info -S 0.2 -t 0.1 -nants 8      # header + size of the selection, nothing read
python DADAnalyse.py file.dada -S 0.2 -t 0.1 -nants 8 -nchans 16 --plot diag ib
python DADAnalyse.py file.dada -t 0.05 --ants 49-56 --chans 0-31 --save section.npy
```

| flag | meaning |
|---|---|
| `-S SEC` | seconds to skip from the start |
| `-t SEC` | seconds to load (default: to the end) |
| `-nants N` / `--ants 49-56,3` | first N antennas / explicit antennas |
| `-nchans N` / `--chans 0-15,32` | first N channels / explicit channels |
| `--tsamp US` | sample time override (default header `TSAMP`, else 4.785 us) |
| `--int8` | keep raw int8 when saving (4x smaller than complex64) |

```python
from DADAnalyse import read_dada
d = read_dada("file.dada", start=0.2, duration=0.1, ants="49-56", nchans=16)
d = read_dada("file.dada", load=False)                  # header only
x = d.read(start=0.5, duration=0.01, ants=[49, 56])     # returns an array
for t0, chunk in d.iter_chunks(chunk=0.1, ants="49,56"):  # stream the whole file
    ...
```

`-S` and `-t` are converted to the nearest whole sample (4.785 us for MeerKAT); nothing outside that range is loaded.
Data are held time-major: `combined_data()` is `(Sample, Antenna, Channel, Pol)` and is what `--save` writes.
`get_data()` returns the `(Time, Antenna, Channel, Inner Time, Pol)` view when the number of samples is a multiple of `INNER_T`
(always true for a whole file), and raises an error otherwise.
Antenna and channel arguments to the analysis methods (`compute_visibilities(49, 56)` etc.) are file indices.

## Caution
The code is currently being tested on DADA files of the format TAFTP , the new feature for slicing the data was made using Claude Model Sonnet 5.5 and should be extensively tested for failure modes before publishing (in testing phase)

## Acknowledgement 
I would like to thank Colin Clark and Matthew Bailes for their valuable contributions to the code, I would like to thank Vivek for providing with test DADA files.
