import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

class Digital_Signal_Information:
    def __init__(self, signal_power, n_bit_mod):
        self._signal_power = signal_power
        self._noise_power = 0.0
        self._n_bit_mod = n_bit_mod
        
    @property
    def signal_power(self): return self._signal_power
    @property
    def noise_power(self): return self._noise_power
    @property
    def n_bit_mod(self): return self._n_bit_mod
    
    @signal_power.setter
    def signal_power(self, value): self._signal_power = value
    @noise_power.setter
    def noise_power(self, value): self._noise_power = value
    @n_bit_mod.setter
    def n_bit_mod(self, value): self._n_bit_mod = value
class ADC:
    def __init__(self, n_bit_adc):
        self._n_bit_adc = n_bit_adc
    @property
    def n_bit_adc(self): return self._n_bit_adc
    @n_bit_adc.setter
    def n_bit_adc(self, value): self._n_bit_adc = value
class Line:
    def __init__(self, loss_coefficient, length):
        self._loss_coefficient = loss_coefficient  
        self._length = length                       
    @property
    def loss_coefficient(self): return self._loss_coefficient
    @property
    def length(self): return self._length
    @loss_coefficient.setter
    def loss_coefficient(self, value): self._loss_coefficient = value
    @length.setter
    def length(self, value): self._length = value
    @property
    def loss(self):
        return self._loss_coefficient * (self._length / 1000.0)
    def noise_generation(self, signal_power):
        return 1e-9 * signal_power * self._length
    def snr_digital(self, signal_power):
        p_tx_dbm = 10 * np.log10(signal_power * 1000)
        noise_pwr = self.noise_generation(signal_power)
        if noise_pwr <= 0:
            return float('inf')    
        p_noise_dbm = 10 * np.log10(noise_pwr * 1000)
        return p_tx_dbm - p_noise_dbm - self.loss
class PCM:
    def __init__(self, dsi, adc, line):
        self._dsi = dsi
        self._adc = adc
        self._line = line
    @property
    def dsi(self): return self._dsi
    @property
    def adc(self): return self._adc
    @property
    def line(self): return self._line
    @dsi.setter
    def dsi(self, value): self._dsi = value
    @adc.setter
    def adc(self, value): self._adc = value
    @line.setter
    def line(self, value): self._line = value
    def snr(self):
        snr_db = self._line.snr_digital(self._dsi.signal_power)
        return 10 ** (snr_db / 10.0)
    def ber_evaluation(self):
        snr_lin = self.snr()
        n_bit = self._dsi.n_bit_mod
        
        if n_bit == 1:
            return 0.5 * erfc(np.sqrt(snr_lin))
        elif n_bit == 2:
            return 0.5 * erfc(np.sqrt(snr_lin / 2.0))
        elif n_bit == 3:
            return (2/3) * erfc(np.sqrt(snr_lin / 3.0)) 
        elif n_bit == 4:
            return (3/4) * erfc(np.sqrt(snr_lin / 10.0))
        else:
            return 1.0
signal_power_W = 1e-3       
alpha = 1                 
lengths_m = np.linspace(10e3, 120e3, 100)
n_bit_adc = 6
ber_th = 10**-2            
modulations = {
    1: 'BPSK',
    2: 'QPSK',
    3: '8QAM',
    4: '16QAM'
}
results_snr_db = {mod: [] for mod in modulations.values()}
results_ber = {mod: [] for mod in modulations.values()}

for n_bit, mod_name in modulations.items():
    dsi = Digital_Signal_Information(signal_power_W, n_bit)
    adc = ADC(n_bit_adc)
    for L in lengths_m:
        line = Line(alpha, L)
        pcm = PCM(dsi, adc, line)
        

        snr_db_val = line.snr_digital(dsi.signal_power)
        ber_val = pcm.ber_evaluation()
        
        results_snr_db[mod_name].append(snr_db_val)
        

        if ber_val < 1e-30:
            ber_val = 1e-30 
            
        results_ber[mod_name].append(np.log10(ber_val))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for mod_name in modulations.values():
    ax1.plot(results_snr_db[mod_name], results_ber[mod_name], label=mod_name)
ax1.axhline(np.log10(ber_th), color='r', linestyle='--', label=f'BER_th = 10^-2')
ax1.set_xlim(left=-1)          
ax1.set_ylim([-30, 0.1])       
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('log10(BER)')
ax1.set_title('log10(BER) vs SNR')
ax1.legend()
ax1.grid(True)
lengths_km = lengths_m / 1000.0
for mod_name in modulations.values():
    ax2.plot(lengths_km, results_ber[mod_name], label=mod_name)
ax2.axhline(np.log10(ber_th), color='r', linestyle='--', label=f'BER_th = 10^-2')
ax2.set_ylim([-30, 0.1]) 
ax2.set_xlabel('Length (km)')
ax2.set_ylabel('log10(BER)')
ax2.set_title('log10(BER) vs Length')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()
print(f"{'Modulation':<10} | {'Min SNR (dB)':<15} | {'Max Length (km)':<15}")
print("-" * 45)
print(f"BER threshold = {ber_th:.2e} , Loss coefficient = {alpha} dB/km")
for mod_name in modulations.values():
    ber_array = np.array(results_ber[mod_name])
    snr_array = np.array(results_snr_db[mod_name])
    valid_indices = ber_array <= np.log10(ber_th)
    if np.any(valid_indices):
        min_snr = np.min(snr_array[valid_indices])
        max_length = lengths_km[np.max(np.where(valid_indices))]
        print(f"{mod_name:<10} | {min_snr:<15.2f} | {max_length:<15.2f}")
    else:
        print(f"{mod_name:<10} | {'N/A':<15} | {'N/A':<15}")
