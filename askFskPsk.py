import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ------------------ Base Class ------------------

class ModulationScheme(ABC):
    def __init__(self, snr_db=10, num_symbols=10, time=1.0, add_noise=True):
        self.snr_db = snr_db
        self.num_symbols = num_symbols
        self.time = np.linspace(0, 1, int(num_symbols * 10), endpoint=False)
        self.add_noise = add_noise
        self.bits = np.random.randint(0, 2, size=self.num_symbols)
        self.signal_power = 1
        self.noise_power = self.signal_power / (10 ** (self.snr_db / 10))

    @abstractmethod
    def modulate(self):
        pass

    def add_awgn_noise(self, signal):
        noise_std = np.sqrt(self.noise_power / 2)
        noisy_signal = signal + noise_std * np.random.randn(len(signal))
        return noisy_signal


# ------------------ Derived Classes ------------------

class BPSKModulation(ModulationScheme):
    def modulate(self):
        samples_per_bit = 10
        nrz = np.where(self.bits == 1, 1, -1)
        modulated_signal = np.repeat(nrz, samples_per_bit)
        carrier = np.cos(2 * np.pi * 50 * time)
        signal = modulated_signal * carrier
        if self.add_noise:
            signal = self.add_awgn_noise(self)
        return signal


class BFSKModulation(ModulationScheme):
    def modulate(self):
        samples_per_bit = 10
        nrz = np.where(self.bits == 1, 1, -1)
        modulated_signal = np.repeat(nrz, samples_per_bit)
        carrier = np.cos(2 * np.pi * 50 * time)
        signal = modulated_signal * carrier
        if self.add_noise:
            signal = self.add_awgn_noise(self)
        return signal


class BASKModulation(ModulationScheme):
    def modulate(self):
        samples_per_bit = 10
        nrz = np.where(self.bits == 1, 1, -1)
        modulated_signal = np.repeat(nrz, samples_per_bit)
        carrier = np.cos(2 * np.pi * 50 * time)
        signal = modulated_signal * carrier
        if self.add_noise:
            signal = self.add_awgn_noise(self)
        return signal


# ------------------ Simulator Class ------------------

class DigitalModulationSimulator:
    def __init__(self, modulation_type="bpsk", snr_db=10, num_symbols=1000, time, add_noise=True):
        self.modulation_type = modulation_type.lower()
        self.snr_db = snr_db
        self.num_symbols = num_symbols
        self.time = time
        self.add_noise = add_noise
        self.modulator = self.get_modulator()

    def get_modulator(self):
        if self.modulation_type == "bpsk":
            return BPSKModulation(self.snr_db, self.num_symbols, self.add_noise)
        elif self.modulation_type == "bfsk":
            return BFSKModulation(self.snr_db, self.num_symbols, self.add_noise)
        elif self.modulation_type == "bask":
            return BASKModulation(self.snr_db, self.num_symbols, self.add_noise)
        else:
            raise ValueError("Unsupported modulation type. Choose from 'bpsk', 'bfsk', or 'bask'.")

    def plot_constellation(self):
        signal = self.modulator.modulate()
        plt.figure(figsize=(6, 6))
        plt.scatter(signal, color='teal', alpha=0.6, s=10)
        plt.title(f'Modulated Signal - {self.modulation_type.upper()} {"(with noise)" if self.add_noise else "(no noise)"}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.show()

# Delete any accidental shadowing
try:
    del input
except:
    pass
# ------------------ Run Example with User Input ------------------

if __name__ == "__main__":
    user_mod_type = input("Enter modulation type (bpsk, bfsk, bask): ").strip().lower()

    if user_mod_type not in ["bpsk", "bfsk", "bask"]:
        raise ValueError(f"Invalid modulation type '{user_mod_type}'. Choose from 'bpsk', 'bfsk', or 'bask'.")

    user_snr = float(input("Enter SNR in dB (e.g., 10, 15): "))
    user_noise_choice = input("Add noise? (yes/no): ").strip().lower()
    add_noise = user_noise_choice == "yes"

    sim = DigitalModulationSimulator(
        modulation_type=user_mod_type,
        snr_db=user_snr,
        add_noise=add_noise
    )

    sim.plot_constellation()

