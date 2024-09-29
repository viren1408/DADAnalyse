import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import resample


class DADAnalyse:
    def __init__(self, file_path):
        self.file_path = file_path
        self.header = None
        self.data = None
        self.load_file()

    def load_file(self):
        # Read header
        with open(self.file_path, 'rb') as f:
            header = f.read(4096).decode('ascii')
        self.header = self.parse_header(header)
        
        # Read  data
        with open(self.file_path, 'rb') as f:
            f.seek(self.header['HDR_SIZE'])
            raw_data = np.fromfile(f, dtype=np.int8)

        if self.header['ORDER'] == 'TAFTP': #if this is not present is this a norm? 
            self.data = np.asarray(
                np.reshape(raw_data, (-1, self.header['NANT'], self.header['NCHAN'], self.header['INNER_T'], self.header['NPOL'], self.header['NDIM'])),dtype='float32').view('complex64').squeeze()

    def parse_header(self, header):
        header_dict = {}
        for line in header.split('\n'):
            if line and not line.startswith('#') and ' ' in line:
                key, value = line.split(None, 1)
                header_dict[key] = value.strip()

        # Cast to appropriate types
        header_dict['NBIT'] = int(header_dict['NBIT'])
        header_dict['NDIM'] = int(header_dict['NDIM'])
        header_dict['NPOL'] = int(header_dict['NPOL'])
        header_dict['NCHAN'] = int(header_dict['NCHAN'])
        header_dict['NANT'] = int(header_dict['NANT'])
        header_dict['INNER_T'] = int(header_dict['INNER_T'])
        header_dict['HDR_SIZE'] = int(header_dict['HDR_SIZE'])
        header_dict['ORDER'] = str(header_dict['ORDER'])
        header_dict['CHAN0_IDX'] = int(header_dict['CHAN0_IDX'])
        header_dict['Lowest_freq (MHz)'] = header_dict['CHAN0_IDX'] * 0.208984375 + 856  # channel_width = 856 / 4096 ; bandwidth=856 MHz

        return header_dict

    def get_header(self):
        """
        Prints the header info 
        """
        return self.header

    def get_data(self):
        """
        Get the volatage data from the dada file in a numpy array 

        data(np.ndarray) : (Time,Antenna,Freqeuncy,Inner Time,Polarisation)
        """
        return self.data

    def plot_antenna_diag(self):
        """
        Plot the power spectrum of each antenna.

        This function calculates average power across all time samples each 
        antenna.This is useful to diagnose if a file is RFI prone and if the RFI is affecting all the antennas in a similar fashion, it also compares the power in both the polarisation to get a clear comparison. 


        - Only the first 57 antennas will be plotted, this is for current convinience of the available data 
        - A PNG file showing the pulse profiles for each antennais saved
        """

        channel_width = 856 / 4096  # MHz
        lowest_frequency = self.header['Lowest_freq (MHz)']
        power = np.abs(self.data) ** 2

        avg_inner_t = np.mean(power, axis=3)
        avg_power = np.mean(avg_inner_t, axis=0)

        nchan = avg_power.shape[1]
        freqs = np.arange(nchan) * channel_width + lowest_frequency

        fig, axes = plt.subplots(8, 8, figsize=(20, 20))

        for ant_index in range(self.header['NANT']):
            ax = axes[ant_index // 8, ant_index % 8]
            ax.plot(freqs, avg_power[ant_index, :, :])
            ax.set_title(f'Ant: {ant_index}', fontsize=8, loc='left')
            ax.set_ylabel('Power', fontsize=8)

        plt.title(f"{self.header['SOURCE']} : {self.header['Lowest_freq (MHz)']}")
        plt.tight_layout()
        plt.savefig(f"{self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_antenna_diagnostic.png")
        print(f"Saved file: {self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_antenna_diagnostic.png") #it is faster to save than to plot 

    def plot_profile_antenna(self):
        """
        Plot the pulse profiles for each antenna.

        This function calculates the sum of power across all time samples and plots the resampled power for each 
        antenna in a grid. Each subplot corresponds to an individual antenna, showing the time-evolution of 
        the pulse profile for that antenna.


        - Only the first 57 antennas will be plotted, this is for current convinience of the available data 
        - A PNG file showing the pulse profiles for each antennais saved.
        """
    
        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        ant_index = 0

        for i in range(8):
            for j in range(8):
                ax = axes[i, j]
                if ant_index < 57:
                    IB_0DM = np.sum(np.abs(self.data[:, ant_index, :, :, :]) ** 2, axis=(1, 3)).reshape(-1)
                    B = resample(IB_0DM, len(IB_0DM) // 256)
                    ax.plot(B)
                    ax.set_title(f'Ant: {ant_index}', fontsize=8, loc='left')
                    ax.set_xlabel('Time', fontsize=8)
                    ax.set_ylabel('Power', fontsize=8)
                    ant_index += 1
                else:
                    ax.axis('off')
        plt.title(f"{self.header['SOURCE']} : {self.header['Lowest_freq (MHz)']}")

        plt.tight_layout()
        plt.savefig(f"{self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_pulse_prof_ant.png")
        print(f"Saved file: {self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_pulse_prof_ant.png")

    def plot_incoherent_beam_pulse(self):
        """
        Plot the incoherent beam pulse profile.

        This function computes the sum of power across all antennas, channels, and polarizations to create an
        incoherent beam. The resulting time-domain pulse profile is plotted, showing the overall power distribution. Try looking at RFI free regions ! 

 
        - The time samples are resampled to reduce the number of points for plotting.
        - The function plots the total power over time.
        - png file showing the incoherent beam pulse profile.
        """

        fig = plt.figure(figsize=(10, 5))
        IB_0DM = np.sum(np.abs(self.data[:, 0:self.header['NANT'], :, :, :]) ** 2, axis=(1, 2, 4)).reshape(-1)
        B = resample(IB_0DM, len(IB_0DM) // 256)
        plt.xlabel('Time Sample')
        plt.ylabel('Power')
        plt.plot(B)
        plt.title(f"{self.header['SOURCE']} : {self.header['Lowest_freq (MHz)']}")
        plt.savefig(f"{self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_sum_pulse_prof.png")
        print(f"Saved file: {self.header['SOURCE']}_{self.header['Lowest_freq (MHz)']}_sum_pulse_prof.png")
    
    def combined_data(self):
        """
        Combine the two time axes
        """
        combined_data = self.data.transpose(0, 3, 1, 2, 4).reshape(-1, self.data.shape[1], self.data.shape[2], self.data.shape[4])

        return combined_data

    def get_bandpass(self, antenna1=49, antenna2=56, polarization=1, chunk_size=int(1024)): #this seems useful for baseline dependence of RFI but currently its just a cute diagnostic tool 
        """
        The bandpass for a given baseline (ant1-ant2) cross-power spectrum.

        args:
            antenna1 (int): Index of the first antenna. default : 49.
            antenna2 (int): Index of the second antenna. default : 56
            polarization (int): Polarization index (0 or 1). Default : 1
            chunk_size (int): Size of chunks for FFT. default :1024 

        returns:
            bandpass (np.ndarray): The combined bandpass for the specified baseline across all channels.
            plots the bandpass for the given baseline 
        """

        combined_data = self.combined_data()

        num_channels = combined_data.shape[2]  # Total number of channels

        data1 = combined_data[:,antenna1,:, :]
        data2 = combined_data[:,antenna2, :,:]

        bandpass = np.zeros((num_channels, chunk_size))
        band = np.zeros((64,1024))

        plt.figure(figsize=(14,6))
        #calculate the bandpass for the cross power in two antennas 
        for channel in range(num_channels):
            data_channel_antenna1 =data1[:,channel,:]
            data_channel_antenna2 = data2[:,channel,:]

            num_samples = data_channel_antenna1.shape[0]
            num_chunks= num_samples // chunk_size

            data_channel_antenna1= data_channel_antenna1[:num_chunks *chunk_size]
            data_channel_antenna2 = data_channel_antenna2[:num_chunks* chunk_size]

            data_chunks_antenna1= data_channel_antenna1.reshape(num_chunks, chunk_size,data_channel_antenna1.shape[1])
            data_chunks_antenna2 =data_channel_antenna2.reshape(num_chunks,chunk_size, data_channel_antenna2.shape[1])

            fft_chunks_antenna1= np.fft.fftshift(np.fft.fft(data_chunks_antenna1, axis=1)) #shift zero frequnency so the x axis is f-f0
            fft_chunks_antenna2 = np.fft.fftshift(np.fft.fft(data_chunks_antenna2, axis=1)) #shift zero frequnency so the x axis is f-f0

            cross_power_spectrum = np.abs(fft_chunks_antenna1 * np.conj(fft_chunks_antenna2))**2

            avg_cross_power_spectrum = np.mean(cross_power_spectrum, axis=0)
            bandpass[channel,:] = (avg_cross_power_spectrum[:, polarization])
            
            plt.plot(np.arange(chunk_size)+ chunk_size*channel,avg_cross_power_spectrum[:, polarization], color='black') #showed some problems when chunk size was decreased works for 1024 


        #band = np.concatenate(bandpass, axis=0)

        plt.xlabel('Frequency Index (across all channels)')
        plt.ylabel('Cross-Power Spectrum (arbitrary units)')
        plt.title(f'Combined Bandpass Across All Channels for Baseline {antenna1}-{antenna2} (Polarization {polarization})')
        plt.grid(True)

        plt.xticks(np.arange(0,1024*num_channels,1024), labels=[f'Ch {i}' for i in range(num_channels)], rotation=90)
        plt.yscale('log')
        plt.show()

        return bandpass

    def compute_visibilities(self, antenna1, antenna2, chunk_size=1024): 
        """
        Compute the visibilities between two antennas using the baseband data.

        This function segments the voltage data into chunks, performs a Fourier transform on each chunk, 
        and then computes the cross-spectral density for the specified antenna pair. 

        Args:
            antenna1 (int): Index of the first antenna.
            antenna2 (int): Index of the second antenna.
            chunk_size (int): Number of time samples to process in each chunk (default: 128).

        Returns:
            cross_spectral_density (np.ndarray): A complex array containing the cross-spectral density of the 
            specified antenna pair. The dimensions are (chunk_size, num_channels, num_polarizations).

        """       

        combined_data = self.combined_data()
        
        data1 = combined_data[:,antenna1, :, :]
        data2 =combined_data[:,antenna2, :, :]
        
        num_chunks = data1.shape[0] // chunk_size
        cross_spectral_density = np.zeros((chunk_size,data1.shape[1], data1.shape[2]),dtype=np.complex64)
        
        for i in range(num_chunks):
            start_idx = i*chunk_size
            end_idx = (i + 1)*chunk_size

            data1_chunk = data1[start_idx:end_idx,:,:]
            data2_chunk =data2[start_idx:end_idx,:, :]

            fft_data1_chunk= np.fft.fftshift(np.fft.fft(data1_chunk, axis=0), axes=0)
            fft_data2_chunk =np.fft.fftshift(np.fft.fft(data2_chunk, axis=0), axes=0)

            cross_spec_chunk =fft_data1_chunk * np.conjugate(fft_data2_chunk)
            cross_spectral_density+= cross_spec_chunk

        return cross_spectral_density
    
    
    def plot_fringes(self,antenna1,antenna2,channel=10, polarization=1):
        """
        Plot the real and complex visibilities to check the existence of fringes and phase wrapping for a given antenna baseline ,frequency channel and polarisation. 

        Args:
            antenna1 (int): Index of antenna1.
            antenna2 (int): Index of antenna2.
            channel (int): Frequency channel index (defualt : 10).

        - Ensure that the antenna indices are within the range of the loaded data.The baseline length code is yet to be tested ! 
        - Ensure the channel is specified within range (most likely : 0-64)
        """       
    
        visibilities = self.compute_visibilities(antenna1,antenna2)

        num_chunks = visibilities.shape[0]
        time = np.arange(num_chunks)
        visibility_real = visibilities[:, channel, polarization].real
        visibility_imag = visibilities[:, channel, polarization].imag
        phase = np.arctan2(visibility_imag, visibility_real)/np.pi

        fig, axs = plt.subplots(3,1, figsize=(12, 10), sharex=True)

        axs[0].plot(time, visibility_real, label='Real Part', color='red')
        axs[0].set_title(f'Visibility Fringes (Channel {channel}, Polarization {polarization})')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Real Part')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(time, visibility_imag,label='Imaginary Part', color='green')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Imaginary Part')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(time, phase, label='Phase', color='blue')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Phase (π)')
        axs[2].legend()
        axs[2].grid(True)

       
        plt.suptitle(f"Lowest Frequency: {self.header['Lowest_freq (MHz)']} MHz", fontsize=16)

        plt.show()


    def analyze_cross_co(self, antenna1,antenna2,plot_snr=False,chunk_size=1024,polarisation=1):

        """
            Analyze the cross-correlation for a given baseline (between two antennas) and calculate the S/N for each channel.

            Args:
                antenna1 (int): Index of the first antenna.
                antenna2 (int): Index of the second antenna.
                plot_snr (bool): Whether to plot the S/N values across channels. Default is False.

            Returns:
                cross_co_max_snr_array (np.ndarray): A 2D array where the first column contains the peak S/N bin index 
                and the second column contains the corresponding S/N values for each channel.
                
                cross_co (np.ndarray): The cross-correlation data for the specified baseline with shape (fft_bins,channels,polarisation)
                
                snr_cross_co (np.ndarray): The cross-correlation data for the specified baseline and polarisation.(fft_bins,channels)

            This function computes the cross-correlation between the specified antennas, extracts the peak S/N value 
            for each channel, and optionally plots the S/N values. The S/N is calculated by comparing the peak value 
            of the cross-correlation against the root mean square (RMS) of the off-peak regions. 
        """

        combined_data = self.combined_data()

        visibilities = self.compute_visibilities(antenna1,antenna2,chunk_size)
        
        cross_co = np.fft.fftshift(np.fft.ifft(visibilities, axis=0))
        
        num_channels = combined_data.shape[2]

        num_pols = combined_data.shape[3]

        snr_cross = np.zeros(num_channels)
        
        cross_co_max_snr_array = np.zeros((num_channels, 2))

        snr_cross_co = np.zeros((chunk_size,num_channels))
        
        for channel in range(num_channels):
            median_sub_cross_co = np.abs(cross_co[:, channel,polarisation]) -np.median(np.abs(cross_co[:, channel,polarisation]))
            peak_cross_co_index = np.argmax(median_sub_cross_co)

            peak_window_size = cross_co.shape[0] // 10 
            rms_window_size = 2 * cross_co.shape[0] // 10  

            if peak_cross_co_index + peak_window_size + rms_window_size < cross_co.shape[0]:
                off_peak_region = np.abs(median_sub_cross_co[peak_cross_co_index + peak_window_size: peak_cross_co_index + peak_window_size +rms_window_size])
            else:
                off_peak_region = np.abs(median_sub_cross_co[peak_cross_co_index - peak_window_size -rms_window_size: peak_cross_co_index -peak_window_size])

            off_rms_median_sub_cross_co = np.sqrt(np.mean(np.square(off_peak_region)))
            snr_cross_co[:,channel] = median_sub_cross_co / off_rms_median_sub_cross_co
            peak_snr_bin = np.argmax(snr_cross_co[:,channel])
            snr_cross[channel] = snr_cross_co[peak_snr_bin,channel]
            
            cross_co_max_snr_array[channel, 0]= peak_snr_bin  # bin index
            cross_co_max_snr_array[channel, 1] =snr_cross_co[peak_snr_bin,channel]  # SNR_peak value
        
        if plot_snr==True:
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(num_channels), snr_cross, label=f'Baseline {antenna1}-{antenna2}', marker='o')
            plt.xlabel('Channel')
            plt.ylabel('S/N')
            plt.title('S/N Across Channels')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        return cross_co_max_snr_array,cross_co,snr_cross_co

def read_dada(file_name):
    return DADAnalyse(file_name)
