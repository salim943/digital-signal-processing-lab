import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ------------------ Base Class ------------------

class ModulationScheme(ABC):
    def __init__(self, snr_db=10, num_symbols=1000, add_noise=True):
        self.snr_db = snr_db
        self.num_symbols = num_symbols
        self.add_noise = add_noise
        self.bits = np.random.randint(0, 2, size=self.num_symbols)
        self.signal_power = 1
        self.noise_power = self.signal_power / (10 ** (self.snr_db / 10))

    @abstractmethod
    def modulate(self):
        pass

    def add_awgn_noise(self, I, Q):
        noise_std = np.sqrt(self.noise_power / 2)
        I_noisy = I + noise_std * np.random.randn(*I.shape)
        Q_noisy = Q + noise_std * np.random.randn(*Q.shape)
        return I_noisy, Q_noisy


# ------------------ Derived Classes ------------------

class BPSKModulation(ModulationScheme):
    def modulate(self):
        I = np.where(self.bits == 1, 1, -1).astype(float)
        Q = np.zeros_like(I, dtype=float)
        if self.add_noise:
            I, Q = self.add_awgn_noise(I, Q)
        return I, Q


class BFSKModulation(ModulationScheme):
    def modulate(self):
        I = np.where(self.bits == 1, 1, 0).astype(float)
        Q = np.where(self.bits == 0, 1, 0).astype(float)
        if self.add_noise:
            I, Q = self.add_awgn_noise(I, Q)
        return I, Q


class BASKModulation(ModulationScheme):
    def modulate(self):
        I = np.where(self.bits == 1, 1, 0).astype(float)
        Q = np.zeros_like(I, dtype=float)
        if self.add_noise:
            noise = np.sqrt(self.noise_power) * np.random.randn(*I.shape)
            I = I + noise
        return I, Q


# ------------------ Simulator Class ------------------

class DigitalModulationSimulator:
    def __init__(self, modulation_type="bpsk", snr_db=10, num_symbols=1000, add_noise=True):
        self.modulation_type = modulation_type.lower()
        self.snr_db = snr_db
        self.num_symbols = num_symbols
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
        I, Q = self.modulator.modulate()
        plt.figure(figsize=(6, 6))
        plt.scatter(I, Q, color='teal', alpha=0.6, s=10)
        plt.title(f'Constellation Diagram - {self.modulation_type.upper()} {"(with noise)" if self.add_noise else "(no noise)"}')
        plt.xlabel('In-phase (I)')
        plt.ylabel('Quadrature (Q)')
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

