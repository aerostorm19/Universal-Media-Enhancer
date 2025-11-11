"""
Universal Media Enhancer

Enhance images and audio using fundamental DSP/DIP algorithms.
Features: Filtering, edge detection, frequency-domain processing, denoising, and quality metrics.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from src.image_processing.filters import gaussian_filter, median_filter, apply_filter_frequency_domain
from src.image_processing.edge_detection import sobel_edge_detection, canny_edge_detection
from src.image_processing.enhancement import histogram_equalization
from src.image_processing.denoising import wiener_filter
from src.metrics.image_metrics import evaluate_image_quality
from src.signal_processing.frequency_analysis import compute_fft, apply_frequency_filter
from src.signal_processing.spectral_analysis import compute_stft, compute_spectrogram
from src.metrics.signal_metrics import evaluate_audio_quality

class MediaEnhancer:
    def __init__(self):
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image(self, image_path, config):
        print("\n[Loading image]")
        img_color = cv2.imread(image_path)
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        processed = img.copy()
        orig = img.copy()
        stages = []

        if config.get('gaussian', True):
            processed = gaussian_filter(processed, config.get('gaussian_kernel', 5), config.get('gaussian_sigma', 1.0))
            stages.append(('Gaussian', processed.copy()))
        if config.get('median', True):
            processed = median_filter(processed, config.get('median_kernel', 5))
            stages.append(('Median', processed.copy()))
        if config.get('fft_filter', True):
            processed = apply_filter_frequency_domain(processed, config.get('fft_type', 'lowpass'), config.get('fft_cutoff', 0.1))
            stages.append(('FFT-Filtered', processed.copy()))
        if config.get('edge_detection', True):
            sobel = sobel_edge_detection(processed)
            stages.append(('Sobel Edges', sobel))
            canny = canny_edge_detection(processed, config.get('canny_low', 50), config.get('canny_high', 150))
            stages.append(('Canny Edges', canny))
        if config.get('hist_eq', True):
            processed = histogram_equalization(processed, config.get('hist_method', 'clahe'))
            stages.append(('HistEq', processed.copy()))
        if config.get('wiener', True):
            processed = wiener_filter(processed, kernel_size=config.get('wiener_kernel', 5))
            stages.append(('Wiener', processed.copy()))

        metrics = evaluate_image_quality(orig, processed)
        print(f"\nPSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        cv2.imwrite(os.path.join(self.output_dir, "enhanced_image.png"), processed)
        self.plot_progress(orig, processed, stages, metrics)
        with open(os.path.join(self.output_dir, "image_metrics.txt"), 'w') as f:
            f.write(str(metrics))

    def plot_progress(self, orig, final, stages, metrics):
        plt.figure(figsize=(15, 10))
        plt.subplot(3,3,1); plt.imshow(orig, cmap='gray'), plt.title("Original"); plt.axis('off')
        for idx, (title, img) in enumerate(stages):
            plt.subplot(3,3,idx+2)
            plt.imshow(img, cmap='gray'); plt.title(title); plt.axis('off')
        plt.subplot(3,3,9)
        plt.axis('off')
        s = f"PSNR: {metrics['psnr']:.2f}\nSSIM: {metrics['ssim']:.3f}\nMSE: {metrics['mse']:.2f}"
        plt.text(0, 0.5, s, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "image_comparison.png"))
        plt.close()

    def process_audio(self, audio_path, config):
        print(f"\n[Loading audio] {audio_path}")
        sig, sr = sf.read(audio_path)
        if sig.ndim > 1: sig = np.mean(sig, axis=1)
        orig = sig.copy()
        current = sig.copy()

        if config.get('fft_analysis', True):
            freq, mag, phase = compute_fft(current, sr)
        if config.get('freq_filter', True):
            if config.get('filter_type', 'bandpass') == 'bandpass':
                cutoff = (config.get('low_cutoff', 300), config.get('high_cutoff', 3400))
            else:
                cutoff = config.get('cutoff', 1000)
            current = apply_frequency_filter(current, sr, config.get('filter_type', 'bandpass'), cutoff)
        if config.get('stft', True):
            freq_stft, t_stft, stft_matrix = compute_stft(current, sr, config.get('n_fft', 2048), config.get('hop_length', 512))
            freq_spec, t_spec, spec = compute_spectrogram(current, sr, config.get('n_fft', 2048), config.get('hop_length', 512))

        metrics = evaluate_audio_quality(orig, current)
        print(f"SNR: {metrics['snr']:.2f} dB | RMS(orig): {metrics['rms_clean']:.4f} | RMS(proc): {metrics['rms_processed']:.4f}")
        sf.write(os.path.join(self.output_dir, "enhanced_audio.wav"), current, sr)
        self.plot_audio(orig, current, sr, freq_stft, t_stft, stft_matrix, metrics)
        with open(os.path.join(self.output_dir, "audio_metrics.txt"), "w") as f:
            f.write(str(metrics))

    def plot_audio(self, orig, proc, sr, freq, t, stft_matrix, metrics):
        import matplotlib.pyplot as plt
        duration = len(orig)/sr
        t_axis = np.linspace(0, duration, len(orig))
        plt.figure(figsize=(16,10))
        plt.subplot(2,2,1)
        plt.plot(t_axis, orig, label="Original"), plt.title("Original (time)")
        plt.subplot(2,2,2)
        plt.plot(t_axis, proc, label="Enhanced"), plt.title("Enhanced (time)")
        plt.subplot(2,2,3)
        plt.pcolormesh(t, freq, np.abs(stft_matrix), shading='gouraud')
        plt.title("STFT (Spec)")
        plt.subplot(2,2,4)
        plt.axis('off')
        plt.text(0, 0.5, f"SNR: {metrics['snr']:.2f}\nRMS(orig):{metrics['rms_clean']:.3f}\nRMS(new):{metrics['rms_processed']:.3f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "audio_comparison.png"))
        plt.close()

def get_user_input():
    print("\nUNIVERSAL MEDIA ENHANCER\n")
    while True:
        c = input("Select media type - (i)mage or (a)udio: ").strip().lower()
        if c in ['i', 'a']: break
    media_type = 'image' if c == 'i' else 'audio'
    file_path = input("Enter path to {} file: ".format(media_type)).strip()

    config = {}
    if media_type == 'image':
        config['gaussian']    = input("Apply Gaussian filter? [Y/n]: ").strip().lower() != 'n'
        if config['gaussian']:
            config['gaussian_kernel'] = int(input("  Kernel Size [5]: ") or "5")
            config['gaussian_sigma']  = float(input("  Sigma [1.0]: ") or "1.0")
        config['median']      = input("Apply Median filter? [Y/n]: ").strip().lower() != 'n'
        if config['median']:
            config['median_kernel'] = int(input("  Kernel Size [5]: ") or "5")
        config['fft_filter']  = input("FFT frequency filter? [Y/n]: ").strip().lower() != 'n'
        if config['fft_filter']:
            config['fft_type']   = input("  (lowpass/highpass) [lowpass]: ") or "lowpass"
            config['fft_cutoff'] = float(input("  Cutoff (0-0.5) [0.1]: ") or "0.1")
        config['edge_detection'] = input("Edge detection? [Y/n]: ").strip().lower() != 'n'
        if config['edge_detection']:
            config['canny_low'] = int(input("  Canny low threshold [50]: ") or "50")
            config['canny_high'] = int(input("  Canny high threshold [150]: ") or "150")
        config['hist_eq']     = input("Histogram equalization? [Y/n]: ").strip().lower() != 'n'
        if config['hist_eq']:
            config['hist_method']   = input("  Method (global/clahe) [clahe]: ") or "clahe"
        config['wiener']      = input("Wiener filter? [Y/n]: ").strip().lower() != 'n'
        if config['wiener']:
            config['wiener_kernel'] = int(input("  Kernel Size [5]: ") or "5")
    else:
        config['fft_analysis']= input("FFT analysis? [Y/n]: ").strip().lower() != 'n'
        config['freq_filter'] = input("Frequency filter? [Y/n]: ").strip().lower() != 'n'
        if config['freq_filter']:
            config['filter_type'] = input("  (lowpass/highpass/bandpass) [bandpass]: ") or "bandpass"
            if config['filter_type'] == 'bandpass':
                config['low_cutoff'] = int(input("    Low cutoff [300]: ") or "300")
                config['high_cutoff']= int(input("    High cutoff [3400]: ") or "3400")
            else:
                config['cutoff'] = int(input("    Cutoff [1000]: ") or "1000")
        config['stft']        = input("STFT/spectrogram? [Y/n]: ").strip().lower() != 'n'
        if config['stft']:
            config['n_fft']    = int(input("  FFT size [2048]: ") or "2048")
            config['hop_length'] = int(input("  Hop length [512]: ") or "512")
    return media_type, file_path, config

def main():
    media_type, file_path, config = get_user_input()
    enhancer = MediaEnhancer()
    if media_type == 'image':
        enhancer.process_image(file_path, config)
    else:
        enhancer.process_audio(file_path, config)
    print("\nProcessing complete. See the 'output/' directory for results.\n")

if __name__ == '__main__':
    main()
