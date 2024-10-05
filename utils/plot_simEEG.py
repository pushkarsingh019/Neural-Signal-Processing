import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

def plot_simEEG(EEG, chan, fignum):
    """
    plot_simEEG - plot function for EEG simulation
    """
    
    if EEG is None:
        raise ValueError("No inputs provided!")

    plt.figure(fignum, figsize=(10, 8))  # Set figure size
    plt.clf()

    # ERP
    ax1 = plt.subplot(211)
    for trial in range(EEG.data.shape[2]):
        ax1.plot(EEG.times, EEG.data[chan-1, :, trial], linewidth=0.5, color=[0.75]*3)
    
    ax1.plot(EEG.times, np.mean(EEG.data[chan-1, :, :], axis=1), 'k', linewidth=3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Activity')
    ax1.set_title(f'ERP from channel {chan}')

    # Static power spectrum
    hz = np.linspace(0, EEG.srate, EEG.pnts)
    
    if EEG.data.ndim == 3:
        pw = np.mean((2 * np.abs(fft(EEG.data[chan-1, :, :], axis=0) / EEG.pnts))**2, axis=1)
    else:
        pw = (2 * np.abs(fft(EEG.data[chan-1, :]) / EEG.pnts))**2
    
    ax2 = plt.subplot(223)
    ax2.plot(hz, pw, linewidth=2)
    ax2.set_xlim([0, 40])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    ax2.set_title('Static Power Spectrum')

    # Time-frequency analysis
    frex = np.linspace(2, 30, 40)
    waves = 2 * (np.linspace(3, 10, len(frex)) / (2 * np.pi * frex))**2
    wavet = np.arange(-2, 2, 1/EEG.srate)
    halfw = len(wavet) // 2 + 0
    nConv = EEG.pnts * EEG.data.shape[2] + len(wavet) - 1

    tf = np.zeros((len(frex), EEG.pnts))

    dataX = fft(EEG.data[chan-1, :, :].reshape(-1), nConv)

    for fi in range(len(frex)):
        waveX = fft(np.exp(2j * np.pi * frex[fi] * wavet) * np.exp(-wavet**2 / waves[fi]), nConv)
        waveX /= np.max(waveX)

        as_signal = ifft(waveX * dataX)
        as_signal = as_signal[halfw-1:-halfw]  # adjusted slice
        as_signal = as_signal.reshape(EEG.pnts, EEG.data.shape[2])

        tf[fi, :] = np.mean(np.abs(as_signal), axis=1)

    print("Shape of tf:", tf.shape)  # Debugging print
    print("Length of EEG.times:", len(EEG.times))  # Debugging print
    print("Length of frex:", len(frex))  # Debugging print

    ax3 = plt.subplot(224)
    try:
        contour = ax3.contourf(EEG.times, frex, tf, 40, cmap='RdYlBu', linestyles='none')
        plt.colorbar(contour, ax=ax3)  # Add a colorbar
    except Exception as e:
        print("Error in contourf:", e)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Time-Frequency Plot')

    plt.tight_layout()  # Adjust the layout
    plt.show()