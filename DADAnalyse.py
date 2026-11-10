"""
DADAnalyse: read, slice and inspect voltage data from .dada files (TAFTP order).

The file is memory-mapped, so only the part you ask for is read from disk:
a time range (start + duration), a set of antennas and a set of channels.

Python:
    from DADAnalyse import read_dada
    d = read_dada("file.dada")                                   # whole file (old behaviour)
    d = read_dada("file.dada", start=0.2, duration=0.1,          # 100 ms starting at 0.2 s,
                  nants=8, nchans=16)                            # first 8 antennas, first 16 chans
    d = read_dada("file.dada", duration=0.05, ants="49-56", chans="10,20-23")

    # header only, nothing loaded; read sections later or stream through the file
    d = read_dada("file.dada", load=False)
    chunk = d.read(start=0.5, duration=0.01, ants=[49, 56])
    for t0, block in d.iter_chunks(chunk=0.1, ants="49,56"):
        ...

    d.combined_data()   # (Sample, Antenna, Channel, Pol), always available, no copy
    d.get_data()        # (Time, Antenna, Channel, Inner Time, Pol) view, when samples % INNER_T == 0

Command line:
    python DADAnalyse.py file.dada --info
    python DADAnalyse.py file.dada -S 0.2 -t 0.1 -nants 8 -nchans 16 --plot diag
    python DADAnalyse.py file.dada -t 0.05 --ants 49-56 --chans 0-31 --save section.npy
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib

# TkAgg needs a display; fall back to a file-only backend on headless machines
if os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample

MEERKAT_CHAN_BW_MHZ = 856 / 4096          # 0.208984375 MHz, L-band F-engine channel width
MEERKAT_LOW_FREQ_MHZ = 856.0


def parse_index_list(spec, n_max, name="index"):
    """
    Turn an index spec into a sorted array of unique ints.

    spec can be None (all), an int, a list/array of ints, or a string like "0-3,7,10-12".
    """
    if spec is None:
        return np.arange(n_max)
    if isinstance(spec, (int, np.integer)):
        idx = [int(spec)]
    elif isinstance(spec, str):
        idx = []
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                idx.extend(range(int(lo), int(hi) + 1))
            else:
                idx.append(int(part))
    else:
        idx = [int(i) for i in spec]
    idx = np.unique(np.asarray(idx, dtype=int))
    if idx.size == 0:
        raise ValueError(f"empty {name} selection")
    if idx[0] < 0 or idx[-1] >= n_max:
        raise ValueError(f"{name} selection {idx.tolist()} out of range 0-{n_max - 1}")
    return idx


def _as_slice(idx):
    """Return a slice if idx is a contiguous run (cheap view on the memmap), else the array."""
    if idx.size == idx[-1] - idx[0] + 1:
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


class DADAnalyse:
    def __init__(self, file_path, start=0.0, duration=None,
                 nants=None, ants=None, nchans=None, chans=None,
                 tsamp=None, load=True, chunk_blocks=32, verbose=True):
        """
        Args:
            file_path (str): path to the .dada file.
            start (float): seconds to skip from the start of the file (-S). Default 0.
            duration (float): seconds to load (-t). Default None = to the end of the file.
            nants (int): load the first nants antennas (-nants). Ignored if ants is given.
            ants (str | list): explicit antenna indices, e.g. "49-56" or [0, 5, 49].
            nchans (int): load the first nchans channels (-nchans). Ignored if chans is given.
            chans (str | list): explicit channel indices within the file, e.g. "0-15,32".
            tsamp (float): sample time in microseconds. Default: TSAMP from the header,
                           otherwise 1 / (856/4096 MHz) = 4.785 us (MeerKAT L-band).
            load (bool): if False only the header is read; use read() / iter_chunks() / load_file().
            chunk_blocks (int): file blocks converted per step while loading
                                (keeps the int8 -> complex64 temporary small).
            verbose (bool): print what is being loaded.

        start and duration are converted to the nearest whole sample (4.785 us for MeerKAT);
        nothing outside that range is loaded. The span loaded is stored in the header as
        SAMP_OFFSET / NSAMP_LOADED and 'T_OFFSET (s)' / 'T_LOADED (s)'.
        """
        self.file_path = file_path
        self.chunk_blocks = int(chunk_blocks)
        self.verbose = verbose
        self.data = None

        self.file_header = self._read_header(tsamp)
        self.header = dict(self.file_header)
        self._open_memmap()

        if load:
            self.load_file(start=start, duration=duration, nants=nants, ants=ants,
                           nchans=nchans, chans=chans)

    # ------------------------------------------------------------------ header / layout

    def _read_header(self, tsamp=None):
        with open(self.file_path, "rb") as f:
            raw = f.read(4096)
            hdr = self.parse_header(raw.decode("ascii", errors="ignore"))
            if hdr["HDR_SIZE"] > 4096:        # header larger than the default 4 kB
                f.seek(0)
                hdr = self.parse_header(f.read(hdr["HDR_SIZE"]).decode("ascii", errors="ignore"))

        if hdr["ORDER"] != "TAFTP":
            raise NotImplementedError(f"ORDER {hdr['ORDER']} not supported (only TAFTP)")
        if hdr["NBIT"] != 8:
            raise NotImplementedError(f"NBIT {hdr['NBIT']} not supported (only 8-bit)")

        if tsamp is not None:
            hdr["TSAMP_USED (us)"] = float(tsamp)
        elif "TSAMP" in hdr:
            hdr["TSAMP_USED (us)"] = float(hdr["TSAMP"])
        else:
            hdr["TSAMP_USED (us)"] = 1.0 / MEERKAT_CHAN_BW_MHZ

        hdr["BLOCK_BYTES"] = (hdr["NANT"] * hdr["NCHAN"] * hdr["INNER_T"]
                              * hdr["NPOL"] * hdr["NDIM"] * hdr["NBIT"] // 8)
        data_bytes = os.path.getsize(self.file_path) - hdr["HDR_SIZE"]
        hdr["NBLOCKS"] = data_bytes // hdr["BLOCK_BYTES"]
        if data_bytes % hdr["BLOCK_BYTES"] and self.verbose:
            print(f"Warning: {data_bytes % hdr['BLOCK_BYTES']} trailing bytes ignored "
                  f"(not a whole TAFTP block)")
        hdr["T_BLOCK (s)"] = hdr["INNER_T"] * hdr["TSAMP_USED (us)"] * 1e-6
        hdr["T_FILE (s)"] = hdr["NBLOCKS"] * hdr["T_BLOCK (s)"]
        return hdr

    def parse_header(self, header):
        header_dict = {}
        for line in header.replace("\x00", "").split("\n"):
            if line and not line.startswith("#") and " " in line:
                key, value = line.split(None, 1)
                header_dict[key] = value.strip()

        header_dict["NBIT"] = int(header_dict["NBIT"])
        header_dict["NDIM"] = int(header_dict["NDIM"])
        header_dict["NPOL"] = int(header_dict["NPOL"])
        header_dict["NCHAN"] = int(header_dict["NCHAN"])
        header_dict["NANT"] = int(header_dict["NANT"])
        header_dict["INNER_T"] = int(header_dict["INNER_T"])
        header_dict["HDR_SIZE"] = int(header_dict["HDR_SIZE"])
        header_dict["ORDER"] = str(header_dict.get("ORDER", "TAFTP"))
        header_dict["CHAN0_IDX"] = int(header_dict["CHAN0_IDX"])
        header_dict["Lowest_freq (MHz)"] = header_dict["CHAN0_IDX"] * MEERKAT_CHAN_BW_MHZ + MEERKAT_LOW_FREQ_MHZ

        return header_dict

    def _open_memmap(self):
        h = self.file_header
        # Nothing is read here: pages are pulled from disk only when sliced
        self._mm = np.memmap(self.file_path, dtype=np.int8, mode="r", offset=h["HDR_SIZE"],
                             shape=(h["NBLOCKS"], h["NANT"], h["NCHAN"],
                                    h["INNER_T"], h["NPOL"], h["NDIM"]))

    # ------------------------------------------------------------------ selection

    def _resolve(self, start=0.0, duration=None, nants=None, ants=None, nchans=None, chans=None):
        """
        Turn time/antenna/channel requests into a sample range and index arrays.

        start and duration are converted to the nearest whole sample (no padding to blocks).
        """
        h = self.file_header
        tsamp = h["TSAMP_USED (us)"] * 1e-6
        nsamp_file = h["NBLOCKS"] * h["INNER_T"]

        start = float(start or 0.0)
        if start < 0:
            raise ValueError("start (-S) must be >= 0")
        s0 = int(round(start / tsamp))
        if s0 >= nsamp_file:
            raise ValueError(f"start {start} s is past the end of the file ({h['T_FILE (s)']:.6f} s)")

        if duration is None:
            s1 = nsamp_file
        else:
            if duration <= 0:
                raise ValueError("duration (-t) must be > 0")
            s1 = s0 + max(1, int(round(float(duration) / tsamp)))
            if s1 > nsamp_file:
                if self.verbose:
                    print(f"Warning: requested end {start + duration:.6f} s is past the end of "
                          f"the file; loading to {h['T_FILE (s)']:.6f} s")
                s1 = nsamp_file

        if ants is None and nants is not None:
            ants = np.arange(min(int(nants), h["NANT"]))
        if chans is None and nchans is not None:
            chans = np.arange(min(int(nchans), h["NCHAN"]))
        ant_idx = parse_index_list(ants, h["NANT"], "antenna")
        chan_idx = parse_index_list(chans, h["NCHAN"], "channel")
        return s0, s1, ant_idx, chan_idx

    def _alloc(self, nsamp, ant_idx, chan_idx, dtype):
        """Time-major output: (Sample, Antenna, Channel, Pol) [+ NDIM axis for int8]."""
        h = self.file_header
        shape = (nsamp, ant_idx.size, chan_idx.size, h["NPOL"])
        if dtype == "int8":
            return np.empty(shape + (h["NDIM"],), dtype=np.int8)
        if h["NDIM"] == 2:
            return np.empty(shape, dtype=np.complex64)
        return np.empty(shape, dtype=np.float32)

    def _fill(self, out, s0, s1, ant_idx, chan_idx, dtype):
        """Copy samples s0:s1 of the selection into out, a few blocks at a time."""
        h = self.file_header
        it = h["INNER_T"]
        a_sel, c_sel = _as_slice(ant_idx), _as_slice(chan_idx)
        b0, b1 = s0 // it, -(-s1 // it)                      # blocks covering [s0, s1)
        for i in range(b0, b1, self.chunk_blocks):
            j = min(i + self.chunk_blocks, b1)
            # slices stay views on the memmap; index arrays copy only this chunk
            raw = self._mm[i:j][:, a_sel][:, :, c_sel]       # (blk, A, C, IT, P, D)
            if dtype == "int8":
                conv = raw
            elif h["NDIM"] == 2:
                conv = np.ascontiguousarray(raw, dtype=np.float32).view(np.complex64)[..., 0]
            else:
                conv = raw[..., 0].astype(np.float32)
            # (blk, A, C, IT, P...) -> (blk*IT, A, C, P...)
            conv = np.moveaxis(conv, 3, 1).reshape((-1,) + conv.shape[1:3] + conv.shape[4:])
            lo, hi = max(s0, i * it), min(s1, j * it)        # samples wanted from this chunk
            out[lo - s0:hi - s0] = conv[lo - i * it:hi - i * it]
        return out

    def estimate_bytes(self, start=0.0, duration=None, nants=None, ants=None,
                       nchans=None, chans=None, dtype="complex64"):
        """Bytes of the selection as stored on disk, and in memory after loading (no data read)."""
        h = self.file_header
        s0, s1, ant_idx, chan_idx = self._resolve(start, duration, nants, ants, nchans, chans)
        n = (s1 - s0) * ant_idx.size * chan_idx.size * h["NPOL"] * h["NDIM"]
        mem = n if dtype == "int8" else n * 4
        return n, mem

    # ------------------------------------------------------------------ reading

    def read(self, start=0.0, duration=None, nants=None, ants=None, nchans=None, chans=None,
             dtype="complex64"):
        """
        Read a section of the file and return it, without touching self.data.

        Returns a time-major array (Sample, Antenna, Channel, Polarisation), complex64
        (or int8 with a trailing NDIM axis if dtype="int8", 4x less memory).
        """
        s0, s1, ant_idx, chan_idx = self._resolve(start, duration, nants, ants, nchans, chans)
        out = self._alloc(s1 - s0, ant_idx, chan_idx, dtype)
        return self._fill(out, s0, s1, ant_idx, chan_idx, dtype)

    def load_file(self, start=0.0, duration=None, nants=None, ants=None, nchans=None, chans=None,
                  dtype="complex64"):
        """
        Load a section into memory and update self.header to describe it.

        The data are held time-major as (Sample, Antenna, Channel, Pol); combined_data()
        returns that directly and get_data() returns the (Time, Antenna, Channel, Inner Time, Pol)
        view when the number of samples is a multiple of INNER_T.

        The original header is kept in self.file_header. In self.header:
            NANT, NCHAN        -> number loaded
            ANT_IDX, CHAN_IDX  -> file indices of what was loaded
            CHAN0_IDX, Lowest_freq (MHz) -> first loaded channel
            FREQS (MHz)        -> frequency of every loaded channel
            SAMP_OFFSET, NSAMP_LOADED -> first sample (from file start) and number of samples
            T_OFFSET (s), T_LOADED (s) -> the same in seconds
        """
        h = self.file_header
        s0, s1, ant_idx, chan_idx = self._resolve(start, duration, nants, ants, nchans, chans)
        tsamp = h["TSAMP_USED (us)"] * 1e-6

        if self.verbose:
            n_mem = self.estimate_bytes(start, duration, nants, ants, nchans, chans, dtype)[1]
            print(f"Loading {self.file_path}\n"
                  f"  time : {s0 * tsamp:.6f} - {s1 * tsamp:.6f} s "
                  f"(samples {s0}-{s1}, {s1 - s0} of {h['NBLOCKS'] * h['INNER_T']})\n"
                  f"  ants : {ant_idx.size} of {h['NANT']}\n"
                  f"  chans: {chan_idx.size} of {h['NCHAN']}\n"
                  f"  memory: {n_mem / 1e9:.3f} GB ({dtype})")

        out = self._alloc(s1 - s0, ant_idx, chan_idx, dtype)
        self.data = self._fill(out, s0, s1, ant_idx, chan_idx, dtype)

        self.ant_idx = ant_idx
        self.chan_idx = chan_idx
        self.sample_range = (s0, s1)
        self.freqs = (h["CHAN0_IDX"] + chan_idx) * MEERKAT_CHAN_BW_MHZ + MEERKAT_LOW_FREQ_MHZ

        hdr = dict(h)
        hdr["FILE_NANT"], hdr["FILE_NCHAN"] = h["NANT"], h["NCHAN"]
        hdr["FILE_CHAN0_IDX"] = h["CHAN0_IDX"]
        hdr["NANT"], hdr["NCHAN"] = int(ant_idx.size), int(chan_idx.size)
        hdr["ANT_IDX"], hdr["CHAN_IDX"] = ant_idx.tolist(), chan_idx.tolist()
        hdr["CHAN0_IDX"] = int(h["CHAN0_IDX"] + chan_idx[0])
        hdr["Lowest_freq (MHz)"] = float(self.freqs[0])
        hdr["FREQS (MHz)"] = self.freqs.tolist()
        hdr["CHANS_CONTIGUOUS"] = bool(isinstance(_as_slice(chan_idx), slice))
        hdr["SAMP_OFFSET"] = s0
        hdr["NSAMP_LOADED"] = s1 - s0
        hdr["T_OFFSET (s)"] = s0 * tsamp
        hdr["T_LOADED (s)"] = (s1 - s0) * tsamp
        self.header = hdr
        return self.data

    def iter_chunks(self, chunk=0.1, start=0.0, duration=None, nants=None, ants=None,
                    nchans=None, chans=None, dtype="complex64"):
        """
        Stream through the file `chunk` seconds at a time, holding only one chunk in memory.

        Yields (t_offset_seconds, data) with data time-major (Sample, Antenna, Channel, Pol),
        each chunk the nearest whole number of samples to `chunk` (the last one may be shorter).
        """
        h = self.file_header
        tsamp = h["TSAMP_USED (us)"] * 1e-6
        s0, s1, ant_idx, chan_idx = self._resolve(start, duration, nants, ants, nchans, chans)
        step = max(1, int(round(chunk / tsamp)))
        for i in range(s0, s1, step):
            j = min(i + step, s1)
            out = self._alloc(j - i, ant_idx, chan_idx, dtype)
            yield i * tsamp, self._fill(out, i, j, ant_idx, chan_idx, dtype)

    # ------------------------------------------------------------------ index helpers

    def _ant(self, ant):
        """File antenna index -> position in the loaded data."""
        pos = np.flatnonzero(self.ant_idx == ant)
        if pos.size == 0:
            raise ValueError(f"antenna {ant} not loaded (loaded: {self.ant_idx.tolist()})")
        return int(pos[0])

    def _chan(self, chan):
        """File channel index -> position in the loaded data."""
        pos = np.flatnonzero(self.chan_idx == chan)
        if pos.size == 0:
            raise ValueError(f"channel {chan} not loaded (loaded: {self.chan_idx.tolist()})")
        return int(pos[0])

    def get_header(self):
        """
        Returns the header of the loaded section (file header: self.file_header)
        """
        return self.header

    def get_data(self):
        """
        Get the voltage data in the original block layout (no copy, a view of the loaded data).

        data(np.ndarray) : (Time, Antenna, Frequency, Inner Time, Polarisation)

        Blocks are counted from the first loaded sample. This layout only exists when the
        number of loaded samples is a multiple of INNER_T; otherwise use combined_data(),
        which is always (Sample, Antenna, Channel, Polarisation).
        """
        it = self.file_header["INNER_T"]
        n = self.data.shape[0]
        if n % it:
            raise ValueError(
                f"{n} samples loaded is not a multiple of INNER_T={it}, so the "
                f"(Time, Antenna, Channel, Inner Time, Pol) layout does not exist; use "
                f"combined_data(), or load a whole number of blocks "
                f"(-t multiple of {self.file_header['T_BLOCK (s)']:.9f} s)")
        d = self.data.reshape((n // it, it) + self.data.shape[1:])
        return np.moveaxis(d, 1, 3)

    # ------------------------------------------------------------------ diagnostics

    def _grid(self, n, ncol=8):
        nrow = max(1, math.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2.5 * nrow), squeeze=False)
        for k in range(n, nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        return fig, axes

    def plot_antenna_diag(self):
        """
        Plot the power spectrum of each loaded antenna.

        This function calculates average power across all time samples each
        antenna. This is useful to diagnose if a file is RFI prone and if the RFI is affecting all the
        antennas in a similar fashion, it also compares the power in both the polarisation to get a clear comparison.

        - A PNG file showing the spectra for each antenna is saved
        """
        avg_power = (np.abs(self.data) ** 2).mean(axis=0)      # (ant, chan, pol)

        nant = self.data.shape[1]
        fig, axes = self._grid(nant)
        for k in range(nant):
            ax = axes[k // 8, k % 8]
            ax.plot(self.freqs, avg_power[k, :, :])
            ax.set_title(f"Ant: {self.ant_idx[k]}", fontsize=8, loc="left")
            ax.set_ylabel("Power", fontsize=8)

        fig.suptitle(f"{self.header['SOURCE']} : {self.header['Lowest_freq (MHz)']}")
        plt.tight_layout()
        out = f"{self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_antenna_diagnostic.png"
        plt.savefig(out)
        print(f"Saved file: {out}")  # it is faster to save than to plot

    def plot_profile_antenna(self):
        """
        Plot the pulse profiles for each loaded antenna.

        This function calculates the sum of power across channels and polarisations and plots the
        resampled power for each antenna in a grid.

        - A PNG file showing the pulse profiles for each antenna is saved.
        """
        nant = self.data.shape[1]
        fig, axes = self._grid(nant)
        for k in range(nant):
            ax = axes[k // 8, k % 8]
            IB_0DM = np.sum(np.abs(self.data[:, k]) ** 2, axis=(1, 2))
            B = resample(IB_0DM, max(1, len(IB_0DM) // 256))
            ax.plot(B)
            ax.set_title(f"Ant: {self.ant_idx[k]}", fontsize=8, loc="left")
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Power", fontsize=8)
        fig.suptitle(f"{self.header['SOURCE']} : {self.header['Lowest_freq (MHz)']}")

        plt.tight_layout()
        out = f"{self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_pulse_prof_ant.png"
        plt.savefig(out)
        print(f"Saved file: {out}")

    def plot_incoherent_beam_pulse(self):
        """
        Plot the incoherent beam pulse profile.

        This function computes the sum of power across all loaded antennas, channels, and polarizations
        to create an incoherent beam. Try looking at RFI free regions !

        - The time samples are resampled to reduce the number of points for plotting.
        - png file showing the incoherent beam pulse profile.
        """
        fig = plt.figure(figsize=(10, 5))
        IB_0DM = np.sum(np.abs(self.data) ** 2, axis=(1, 2, 3))
        B = resample(IB_0DM, max(1, len(IB_0DM) // 256))
        plt.xlabel("Time Sample")
        plt.ylabel("Power")
        plt.plot(B)
        plt.title(f"{self.header['SOURCE']} : {self.header['Lowest_freq (MHz)']}")
        out = f"{self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_sum_pulse_prof.png"
        plt.savefig(out)
        print(f"Saved file: {out}")

    def combined_data(self):
        """
        The 2 time axes combined -> (Sample, Antenna, Channel, Pol).
        This is how the data are held, so no copy is made.
        """
        return self.data

    def get_bandpass(self, antenna1=49, antenna2=56, polarization=1, chunk_size=int(1024)):
        """
        The bandpass for a given baseline (ant1-ant2) cross-power spectrum.

        args:
            antenna1 (int): File index of the first antenna. default : 49.
            antenna2 (int): File index of the second antenna. default : 56
            polarization (int): Polarization index (0 or 1). Default : 1
            chunk_size (int): Size of chunks for FFT. default :1024

        returns:
            bandpass (np.ndarray): The combined bandpass for the specified baseline across all loaded channels.
            plots the bandpass for the given baseline
        """
        combined_data = self.combined_data()
        num_channels = combined_data.shape[2]

        data1 = combined_data[:, self._ant(antenna1), :, :]
        data2 = combined_data[:, self._ant(antenna2), :, :]

        bandpass = np.zeros((num_channels, chunk_size))

        plt.figure(figsize=(14, 6))
        for channel in range(num_channels):
            data_channel_antenna1 = data1[:, channel, :]
            data_channel_antenna2 = data2[:, channel, :]

            num_samples = data_channel_antenna1.shape[0]
            num_chunks = num_samples // chunk_size

            data_channel_antenna1 = data_channel_antenna1[:num_chunks * chunk_size]
            data_channel_antenna2 = data_channel_antenna2[:num_chunks * chunk_size]

            data_chunks_antenna1 = data_channel_antenna1.reshape(num_chunks, chunk_size, data_channel_antenna1.shape[1])
            data_chunks_antenna2 = data_channel_antenna2.reshape(num_chunks, chunk_size, data_channel_antenna2.shape[1])

            fft_chunks_antenna1 = np.fft.fftshift(np.fft.fft(data_chunks_antenna1, axis=1), axes=1)
            fft_chunks_antenna2 = np.fft.fftshift(np.fft.fft(data_chunks_antenna2, axis=1), axes=1)

            cross_power_spectrum = np.abs(fft_chunks_antenna1 * np.conj(fft_chunks_antenna2)) ** 2

            avg_cross_power_spectrum = np.mean(cross_power_spectrum, axis=0)
            bandpass[channel, :] = avg_cross_power_spectrum[:, polarization]

            plt.plot(np.arange(chunk_size) + chunk_size * channel, avg_cross_power_spectrum[:, polarization], color="black")

        plt.xlabel("Frequency Index (across all channels)")
        plt.ylabel("Cross-Power Spectrum (arbitrary units)")
        plt.title(f"Combined Bandpass Across All Channels for Baseline {antenna1}-{antenna2} (Polarization {polarization})")
        plt.grid(True)

        plt.xticks(np.arange(0, chunk_size * num_channels, chunk_size),
                   labels=[f"Ch {c}" for c in self.chan_idx], rotation=90)
        plt.yscale("log")
        plt.show()

        return bandpass

    def compute_visibilities(self, antenna1, antenna2, chunk_size=1024):
        """
        Compute the visibilities between two antennas using the baseband data.

        This function segments the voltage data into chunks, performs a Fourier transform on each chunk,
        and then computes the cross-spectral density for the specified antenna pair.

        Args:
            antenna1 (int): File index of the first antenna (must be loaded).
            antenna2 (int): File index of the second antenna (must be loaded).
            chunk_size (int): Number of time samples to process in each chunk (default: 1024).

        Returns:
            cross_spectral_density (np.ndarray): A complex array containing the cross-spectral density of the
            specified antenna pair. The dimensions are (chunk_size, num_channels, num_polarizations).
        """
        combined_data = self.combined_data()

        data1 = combined_data[:, self._ant(antenna1), :, :]
        data2 = combined_data[:, self._ant(antenna2), :, :]

        num_chunks = data1.shape[0] // chunk_size
        cross_spectral_density = np.zeros((chunk_size, data1.shape[1], data1.shape[2]), dtype=np.complex64)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size

            data1_chunk = data1[start_idx:end_idx, :, :]
            data2_chunk = data2[start_idx:end_idx, :, :]

            fft_data1_chunk = np.fft.fftshift(np.fft.fft(data1_chunk, axis=0), axes=0)
            fft_data2_chunk = np.fft.fftshift(np.fft.fft(data2_chunk, axis=0), axes=0)

            cross_spectral_density += fft_data1_chunk * np.conjugate(fft_data2_chunk)

        return cross_spectral_density

    def plot_fringes(self, antenna1, antenna2, channel=10, polarization=1):
        """
        Plot the real and complex visibilities to check the existence of fringes and phase wrapping for a
        given antenna baseline, frequency channel and polarisation.

        Args:
            antenna1 (int): File index of antenna1.
            antenna2 (int): File index of antenna2.
            channel (int): File channel index (default : 10), must be loaded.
        """
        visibilities = self.compute_visibilities(antenna1, antenna2)
        c = self._chan(channel)

        num_chunks = visibilities.shape[0]
        time = np.arange(num_chunks)
        visibility_real = visibilities[:, c, polarization].real
        visibility_imag = visibilities[:, c, polarization].imag
        phase = np.arctan2(visibility_imag, visibility_real) / np.pi

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axs[0].plot(time, visibility_real, label="Real Part", color="red")
        axs[0].set_title(f"Visibility Fringes (Channel {channel}, Polarization {polarization})")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Real Part")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(time, visibility_imag, label="Imaginary Part", color="green")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Imaginary Part")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(time, phase, label="Phase", color="blue")
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Phase (π)")
        axs[2].legend()
        axs[2].grid(True)

        plt.suptitle(f"Channel frequency: {self.freqs[c]} MHz", fontsize=16)
        plt.show()

    def analyze_cross_co(self, antenna1, antenna2, plot_snr=False, chunk_size=1024, polarisation=1):
        """
        Analyze the cross-correlation for a given baseline (between two antennas) and calculate the S/N for each channel.

        Args:
            antenna1 (int): File index of the first antenna.
            antenna2 (int): File index of the second antenna.
            plot_snr (bool): Whether to plot the S/N values across channels. Default is False.

        Returns:
            cross_co_max_snr_array (np.ndarray): (channels, 2): peak S/N bin index and S/N for each loaded channel.
            cross_co (np.ndarray): cross-correlation for the baseline, shape (fft_bins, channels, polarisation)
            snr_cross_co (np.ndarray): S/N of the cross-correlation for the chosen polarisation, (fft_bins, channels)

        The S/N is the median-subtracted peak over the RMS of an off-peak region.
        """
        visibilities = self.compute_visibilities(antenna1, antenna2, chunk_size)

        cross_co = np.fft.fftshift(np.fft.ifft(visibilities, axis=0))

        num_channels = self.data.shape[2]

        snr_cross = np.zeros(num_channels)
        cross_co_max_snr_array = np.zeros((num_channels, 2))
        snr_cross_co = np.zeros((chunk_size, num_channels))

        for channel in range(num_channels):
            median_sub_cross_co = np.abs(cross_co[:, channel, polarisation]) - np.median(np.abs(cross_co[:, channel, polarisation]))
            peak_cross_co_index = np.argmax(median_sub_cross_co)

            peak_window_size = cross_co.shape[0] // 10
            rms_window_size = 2 * cross_co.shape[0] // 10

            if peak_cross_co_index + peak_window_size + rms_window_size < cross_co.shape[0]:
                off_peak_region = np.abs(median_sub_cross_co[peak_cross_co_index + peak_window_size: peak_cross_co_index + peak_window_size + rms_window_size])
            else:
                off_peak_region = np.abs(median_sub_cross_co[peak_cross_co_index - peak_window_size - rms_window_size: peak_cross_co_index - peak_window_size])

            off_rms_median_sub_cross_co = np.sqrt(np.mean(np.square(off_peak_region)))
            snr_cross_co[:, channel] = median_sub_cross_co / off_rms_median_sub_cross_co
            peak_snr_bin = np.argmax(snr_cross_co[:, channel])
            snr_cross[channel] = snr_cross_co[peak_snr_bin, channel]

            cross_co_max_snr_array[channel, 0] = peak_snr_bin
            cross_co_max_snr_array[channel, 1] = snr_cross_co[peak_snr_bin, channel]

        if plot_snr:
            plt.figure(figsize=(10, 6))
            plt.plot(self.chan_idx, snr_cross, label=f"Baseline {antenna1}-{antenna2}", marker="o")
            plt.xlabel("Channel")
            plt.ylabel("S/N")
            plt.title("S/N Across Channels")
            plt.grid(True)
            plt.legend()
            plt.show()

        return cross_co_max_snr_array, cross_co, snr_cross_co


def read_dada(file_name, **kwargs):
    """
    read_dada(file) loads the whole file; keyword arguments are passed to DADAnalyse
    (start, duration, nants, ants, nchans, chans, tsamp, load, ...).
    """
    return DADAnalyse(file_name, **kwargs)


# ---------------------------------------------------------------------- command line

def _cli(argv=None):
    p = argparse.ArgumentParser(
        description="Read a section of a TAFTP .dada voltage file without loading all of it.")
    p.add_argument("file", help=".dada file")
    p.add_argument("-S", "--skip", type=float, default=0.0, metavar="SEC",
                   help="seconds to skip from the start of the file (default 0)")
    p.add_argument("-t", "--time", type=float, default=None, metavar="SEC",
                   help="seconds to load (default: to the end of the file)")
    p.add_argument("-nants", "--nants", type=int, default=None,
                   help="load the first N antennas")
    p.add_argument("--ants", default=None,
                   help='explicit antennas, e.g. "49-56" or "0,5,49" (overrides -nants)')
    p.add_argument("-nchans", "--nchans", type=int, default=None,
                   help="load the first N channels")
    p.add_argument("--chans", default=None,
                   help='explicit channels within the file, e.g. "0-15,32" (overrides -nchans)')
    p.add_argument("--tsamp", type=float, default=None,
                   help="sample time in us (default: header TSAMP, else 4.785 us)")
    p.add_argument("--info", action="store_true",
                   help="print the header and the size of the selection, read no data")
    p.add_argument("--int8", action="store_true",
                   help="keep raw int8 (4x less memory) instead of complex64; only with --save")
    p.add_argument("--save", default=None, metavar="OUT.npy",
                   help="save the selected section to a .npy file")
    p.add_argument("--plot", nargs="+", choices=["diag", "profile", "ib"], default=[],
                   help="diagnostic plots: diag (antenna spectra), profile (per-antenna), ib (incoherent beam)")
    a = p.parse_args(argv)

    if a.int8 and a.plot:
        p.error("--int8 cannot be combined with --plot")

    sel = dict(start=a.skip, duration=a.time, nants=a.nants, ants=a.ants,
               nchans=a.nchans, chans=a.chans)
    d = DADAnalyse(a.file, tsamp=a.tsamp, load=False)
    h = d.file_header

    if a.info:
        for k in ("SOURCE", "UTC_START", "NANT", "NCHAN", "INNER_T", "NPOL", "NDIM", "NBIT",
                  "CHAN0_IDX", "Lowest_freq (MHz)", "TSAMP_USED (us)", "T_BLOCK (s)",
                  "NBLOCKS", "T_FILE (s)"):
            if k in h:
                print(f"{k:20s} {h[k]}")
        b0, b1, ai, ci = d._resolve(**sel)
        n_disk, n_mem = d.estimate_bytes(dtype="int8" if a.int8 else "complex64", **sel)
        ts = h["TSAMP_USED (us)"] * 1e-6
        print(f"\nSelection: {b0 * ts:.6f} - {b1 * ts:.6f} s (samples {b0}-{b1}, {b1 - b0}), "
              f"{ai.size} ants, {ci.size} chans")
        print(f"  reads {n_disk / 1e9:.3f} GB from disk, needs {n_mem / 1e9:.3f} GB in memory")
        return d

    data = d.load_file(dtype="int8" if a.int8 else "complex64", **sel)
    print(f"Loaded array shape {data.shape} {data.dtype}")

    if a.save:
        np.save(a.save, data)
        print(f"Saved {a.save}")
    if "diag" in a.plot:
        d.plot_antenna_diag()
    if "profile" in a.plot:
        d.plot_profile_antenna()
    if "ib" in a.plot:
        d.plot_incoherent_beam_pulse()
    return d


if __name__ == "__main__":
    _cli()
