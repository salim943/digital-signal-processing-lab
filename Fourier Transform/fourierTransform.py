import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class FourierTransformScheme(ABC):
    def __init__(self, samplingFrequency, time, messageFrequency):
        self.samplingFrequency = samplingFrequency
        self.time = time
        self.messageFrequency = messageFrequency

    @abstractmethod
    def FourierTransform(self):
        pass

    # instance method with self
    def compute_fft(self, signal):
        fft_result = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), d=1/self.samplingFrequency)
        return np.abs(fft_result), freq

class sineFourierTransform(FourierTransformScheme):
    def FourierTransform(self):
        signal = np.sin(2 * np.pi * self.messageFrequency * self.time)
        # call compute_fft via self
        return self.compute_fft(signal)

class cosineFourierTransform(FourierTransformScheme):
    def FourierTransform(self):
        signal = np.cos(2 * np.pi * self.messageFrequency * self.time)
        return self.compute_fft(signal)



class rectangularFourierTransform(FourierTransformScheme):
    def FourierTransform(self):
        # Generate a rectangular pulse (value 1 for a limited duration, else 0)
        pulse_width = 1 / self.messageFrequency  # adjust width based on frequency
        signal = np.where(np.abs(self.time) < pulse_width, 1.0, 0.0)
        return self.compute_fft(signal)

class triangularFourierTransform(FourierTransformScheme):
    def FourierTransform(self):
        # Generate a triangular pulse centered at t = 0
        pulse_width = 1 / (2*self.messageFrequency)  # control width with frequency
        signal = np.where(
            np.abs(self.time) < pulse_width,
            1 - (np.abs(self.time) / pulse_width),
            0.0
        )
        return self.compute_fft(signal)


class FourierTransformSimulator:
    def __init__(self, samplingFrequency, messageFrequency, signal_type="sine"):
        self.signal_type = signal_type.lower()
        self.samplingFrequency = samplingFrequency
        self.messageFrequency = messageFrequency
        self.time = np.linspace(0, 1, int(samplingFrequency), endpoint=False)
        self.transformer = self.get_transformer()

    def get_transformer(self):
        if self.signal_type == "sine":
            return sineFourierTransform(self.samplingFrequency, self.time, self.messageFrequency)
        elif self.signal_type == "cosine":
            return cosineFourierTransform(self.samplingFrequency, self.time, self.messageFrequency)
        elif self.signal_type == "rectangular":
            return rectangularFourierTransform(self.samplingFrequency, self.time, self.messageFrequency)
        elif self.signal_type == "triangular":
            return triangularFourierTransform(self.samplingFrequency, self.time, self.messageFrequency)
        else:
            raise ValueError("Unsupported signal type. Choose 'sine' or 'cosine'.")

    def plot_fft(self):
        fft_magnitude, freq = self.transformer.FourierTransform()
        half_len = len(freq) // 2
        plt.figure(figsize=(8, 5))
        plt.plot(freq[:half_len], fft_magnitude[:half_len], color='darkorange')
        plt.title(f'FFT of {self.signal_type.upper()} Wave')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    user_mod_type = input("Enter signal type (sine, cosine, rectangular, triangular): ").strip().lower()
    if user_mod_type not in ["sine", "cosine", "rectangular", "triangular"]:
        raise ValueError(f"Invalid signal type '{user_mod_type}'. Choose 'sine' or 'cosine'.")

    user_sfreq = float(input("Enter sampling frequency (e.g., 1000): "))
    user_mfreq = float(input("Enter message frequency (e.g., 50) or 1 / pulseWidth for reatangular and triangular pulses: "))

    sim = FourierTransformSimulator(
        signal_type=user_mod_type,
        samplingFrequency=user_sfreq,
        messageFrequency=user_mfreq
    )

    sim.plot_fft()