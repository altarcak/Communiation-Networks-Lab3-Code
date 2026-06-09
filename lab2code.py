import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
%matplotlib inline

color_dict = mcol.TABLEAU_COLORS


def lin2db(x):
    return 10*np.log10(x)


def db2lin(x):
    return 10**(x/10)


class ADC:
    fs_step = 2.75625e3
    
    def __init__(self, n_bit):
        self.n_bit = n_bit

    def snr(self):
        return 2 ** (2 * self.n_bit)
    
    def get_sampling_freq(self, bandwidth):
        nyquistr = 2*bandwidth
        
        mltp = np.ceil(nyquistr / self.fs_step)
        return mltp *self.fs_step

    
class BSC:
    
    def __init__(self, error_probability):
        self.error_probability = error_probability
        
    def snr(self):
        return 1 / (4*self.error_probability)


class PCM:

    def __init__(self, n_bit, error_probability, analog_bandwidth=4000.0):
        self.analog_bandwidth = float(analog_bandwidth)
        
        self.adc = ADC(n_bit)
        self.bsc = BSC(error_probability)
    
    def snr(self):
        return (1/self.adc.snr() + 1/self.bsc.snr())**-1
    
    def critical_pe(self):
        return 1 / (4*self.adc.snr())
    
    
def exercise_1():
    n_bit_array = np.array([2, 3, 4, 6, 8, 10, 12, 14, 16], dtype = np.int64)
    
    Pe_array = np.logspace(-12, 0, 100)

    my_adc = ADC(n_bit_array)
    my_bsc = BSC(Pe_array)
    
    snr_q_db = lin2db(my_adc.snr())
    snr_bsc_db = lin2db(my_bsc.snr())
    
    print(f"--- EXERCISE 1 RESULTS ---")
    
    plt.figure(figsize=(12, 5))
    
    #first graphic
    plt.subplot(1, 2, 1)
    plt.plot(n_bit_array, snr_q_db, marker='o', color=color_dict['tab:blue'])
    plt.title("ADC Quantization SNR")
    plt.xlabel("Number of Bits (n_bit)")
    plt.ylabel("SNR_Q (db)")
    plt.grid(True)
    
    #second graphic
    plt.subplot(1, 2, 2)
    plt.plot(Pe_array, snr_bsc_db, color=color_dict['tab:orange'])
    plt.title("BSC Channel SNR")
    plt.xlabel("Error Probability (Pe)")
    plt.ylabel("SNR_BSC (db)")
    plt.xscale("log")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()   
    
    
def exercise_2():
    n_bit_array = [2, 4, 8, 16]
    Pe_array = np.logspace(-12, 0, 100)
    
    print(f"--- EXERCISE 2 RESULTS ---")
    
    plt.figure(figsize=(10, 6))

    
    for n in n_bit_array:
        my_pcm = PCM(n, Pe_array)
        
        snr_tot_db = lin2db(my_pcm.snr())
        line, = plt.plot(Pe_array, snr_tot_db, label=f'n_bit = {n}', linewidth=2)
        c = line.get_color()
        
        snr_q_db = lin2db(my_pcm.adc.snr())
        plt.axhline(snr_q_db, color=c, linestyle='--', alpha=0.7)
        
        pe_th = my_pcm.critical_pe()
        plt.axvline(pe_th, color=c, linestyle=':', alpha=0.7)
        

    plt.xscale("log") 
    plt.title("Exercise 2: Overall PCM System SNR vs Channel Error Probability")
    plt.xlabel("Error Probability (Pe) - Log Scale")
    plt.ylabel("Overall SNR_TOT (dB)")
    plt.grid(True)
    plt.legend() 
    plt.show()
        

def exercise_3():
    bandwidth = 22e3
    target_snr_db = 80
    Pe = 3.8e-7
    
    min_n = target_snr_db / (20 * np.log10(2))
    n_bit = int(np.ceil(min_n))

    my_pcm = PCM(n_bit, Pe, bandwidth)
    
    print(f"--- EXERCISE 3 RESULTS ---")
    print(f"Obtained n_bits of the ADC: {n_bit}")
    
    actual_snr_q_db = lin2db(my_pcm.adc.snr())
    print(f"Effective Quantization SNR: {actual_snr_q_db:.2f} dB")
    
    pe_th = my_pcm.critical_pe()
    print(f"Critical Error Probability (P_th): {pe_th:.2e}")
    print(f"Given BSC Error Probability (Pe): {Pe:.2e}")
    
    if Pe < pe_th:
        print("Conclusion: The BSC CAN support the transmission (Pe is lower than P_th).")
    else:
        print("Conclusion: The BSC CANNOT support the transmission (Pe is higher than P_th).")
        
    
    fs = my_pcm.adc.get_sampling_freq(bandwidth)
    print(f"ADC Sampling Frequency: {fs/1000:.1f} kHz")
    print("-" * 26)


if __name__ == "__main__":
    print("Laboratory 3 - ADC-BSC")
    
    exercise_1()
    exercise_2()
    exercise_3()
