"""
Universal Media Enhancer

Enhance images and audio using fundamental DSP/DIP algorithms.
Features: Filtering, edge detection, frequency-domain processing, denoising, and quality metrics.

Author: Abhijit Rai
Email: abhijit.airosys@gmail.com
"""

import os
import sys
from typing import Dict, Tuple, Any

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

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensures input image is grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def safe_save_image(path: str, img: np.ndarray) -> None:
    """Safely saves image to disk."""
    cv2.imwrite(path, img)


def print_header(title: str) -> None:
    """Prints a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def print_step(step: int, total: int, description: str) -> None:
    """Prints processing step information."""
    print(f"[{step}/{total}] {description}")


def process_image(image_path: str, config: Dict[str, Any]) -> None:
    """
    Process image through enhancement pipeline.
    
    Args:
        image_path: Path to input image
        config: Dictionary containing processing parameters
    """
    print_header("IMAGE PROCESSING PIPELINE")
    
    print_step(1, 8, f"Loading image: {image_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise ValueError(f"Unsupported or corrupt image file: {image_path}")
    
    img = ensure_grayscale(img_color)
    print(f"      Image size: {img.shape}")
    
    processed = img.copy()
    original = img.copy()
    stages = [("Original", img.copy())]

    # Step 2: Gaussian Filter
    if config.get("gaussian", True):
        print_step(2, 8, "Applying Gaussian filter for noise reduction...")
        print(f"      Parameters: kernel_size={config.get('gaussian_kernel', 5)}, sigma={config.get('gaussian_sigma', 1.0)}")
        print(f"      Purpose: Remove high-frequency noise while preserving edges")
        processed = gaussian_filter(
            processed,
            config.get("gaussian_kernel", 5),
            config.get("gaussian_sigma", 1.0)
        )
        stages.append(("Gaussian", processed.copy()))
        print(f"      ✓ Gaussian smoothing complete\n")

    # Step 3: Median Filter
    if config.get("median", True):
        print_step(3, 8, "Applying Median filter for salt-and-pepper noise...")
        print(f"      Parameters: kernel_size={config.get('median_kernel', 5)}")
        print(f"      Purpose: Excellent for impulse noise, preserves edges better than Gaussian")
        processed = median_filter(
            processed,
            config.get("median_kernel", 5)
        )
        stages.append(("Median", processed.copy()))
        print(f"      ✓ Median filtering complete\n")

    # Step 4: FFT Filtering
    if config.get("fft_filter", False):
        print_step(4, 8, "Applying FFT-based frequency domain filtering...")
        print(f"      Parameters: type={config.get('fft_type', 'lowpass')}, cutoff={config.get('fft_cutoff', 0.1)}")
        print(f"      Purpose: Remove specific frequency components (global approach)")
        processed = apply_filter_frequency_domain(
            processed,
            config.get("fft_type", "lowpass"),
            config.get("fft_cutoff", 0.1)
        )
        stages.append(("FFT-Filtered", processed.copy()))
        print(f"      ✓ Frequency filtering complete\n")

    # Step 5: Edge Detection
    if config.get("edge_detection", False):
        print_step(5, 8, "Performing edge detection (Sobel & Canny)...")
        print(f"      Purpose: Identify object boundaries and important features")
        sobel = sobel_edge_detection(processed)
        stages.append(("Sobel Edges", sobel))
        print(f"      ✓ Sobel edge detection complete")
        
        print(f"      Parameters: low_threshold={config.get('canny_low', 50)}, high_threshold={config.get('canny_high', 150)}")
        canny = canny_edge_detection(
            processed,
            config.get("canny_low", 50),
            config.get("canny_high", 150),
            aperture_size=3
        )
        stages.append(("Canny Edges", canny))
        print(f"      ✓ Canny edge detection complete\n")

    # Step 6: Histogram Equalization
    if config.get("hist_eq", False):
        print_step(6, 8, "Applying Histogram Equalization for contrast enhancement...")
        print(f"      Method: {config.get('hist_method', 'clahe').upper()}")
        print(f"      Purpose: Improve contrast by redistributing pixel intensities")
        processed = histogram_equalization(
            processed,
            method=config.get("hist_method", "clahe")
        )
        stages.append(("Histogram Eq", processed.copy()))
        print(f"      ✓ Histogram equalization complete\n")

    # Step 7: Wiener Filter
    if config.get("wiener", False):
        print_step(7, 8, "Applying Wiener filter for advanced denoising...")
        print(f"      Parameters: kernel_size={config.get('wiener_kernel', 5)}")
        print(f"      Purpose: Optimal MSE-based noise reduction")
        processed = wiener_filter(
            processed,
            kernel_size=config.get("wiener_kernel", 5)
        )
        stages.append(("Wiener", processed.copy()))
        print(f"      ✓ Wiener denoising complete\n")

    # Step 8: Quality Metrics
    print_step(8, 8, "Calculating image quality metrics...")
    print(f"      Comparing: Final processed vs Original\n")
    metrics = evaluate_image_quality(original, processed)
    
    print(f"      📊 QUALITY METRICS:")
    print(f"      ├─ PSNR (Peak Signal-to-Noise Ratio): {metrics['psnr']:.2f} dB")
    print(f"      │  Higher is better (typically 20-40 dB is good)")
    print(f"      ├─ SSIM (Structural Similarity Index): {metrics['ssim']:.4f}")
    print(f"      │  Range [0,1], closer to 1 is better")
    print(f"      ├─ MSE (Mean Squared Error): {metrics['mse']:.2f}")
    print(f"      └─ MAE (Mean Absolute Error): {metrics['mae']:.2f}\n")

    # Save Results
    print_header("SAVING RESULTS")
    enhanced_path = os.path.join(OUTPUT_DIR, "enhanced_image.png")
    safe_save_image(enhanced_path, processed)
    print(f"✓ Saved enhanced image: {enhanced_path}")
    
    save_image_results_plot(stages, metrics)
    print(f"✓ Saved comparison: {os.path.join(OUTPUT_DIR, 'image_comparison.png')}")
    
    with open(os.path.join(OUTPUT_DIR, "image_metrics.txt"), 'w') as f:
        f.write("IMAGE QUALITY METRICS\n")
        f.write("="*50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key.upper()}: {value:.4f}\n")
    print(f"✓ Saved metrics: {os.path.join(OUTPUT_DIR, 'image_metrics.txt')}")


def save_image_results_plot(stages: list, metrics: dict) -> None:
    """Save comprehensive image comparison plot."""
    plt.figure(figsize=(18, 10))
    n_stages = len(stages)
    rows, cols = (n_stages + 2) // 3 + 1, 3
    
    for idx, (title, img) in enumerate(stages):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis("off")
    
    plt.subplot(rows, cols, n_stages + 1)
    plt.axis('off')
    metrics_text = f"PSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.3f}\nMSE: {metrics['mse']:.2f}\nMAE: {metrics['mae']:.2f}"
    plt.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "image_comparison.png")
    plt.savefig(outpath, dpi=150)
    plt.close()


def process_audio(audio_path: str, config: Dict[str, Any]) -> None:
    """
    Process audio through enhancement pipeline.
    
    Args:
        audio_path: Path to input audio
        config: Dictionary containing processing parameters
    """
    print_header("AUDIO PROCESSING PIPELINE")
    
    print_step(1, 6, f"Loading audio: {audio_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    sig, sr = sf.read(audio_path)
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)
    
    print(f"      Sample rate: {sr} Hz")
    print(f"      Duration: {len(sig)/sr:.2f} seconds")
    print(f"      Samples: {len(sig)}\n")

    orig = sig.copy()
    current = sig.copy()
    freq_stft, t_stft, stft_matrix = None, None, None

    # Step 2: FFT Analysis
    if config.get('fft_analysis', True):
        print_step(2, 6, "Performing FFT analysis...")
        print(f"      Purpose: Analyze frequency content of the signal")
        freq, mag, phase = compute_fft(current, sr)
        print(f"      ✓ FFT analysis complete\n")

    # Step 3: Frequency Filtering
    if config.get('freq_filter', False):
        print_step(3, 6, "Applying frequency domain filtering...")
        filter_type = config.get('filter_type', 'bandpass')
        if filter_type == 'bandpass':
            cutoff = (config.get('low_cutoff', 300), config.get('high_cutoff', 3400))
            print(f"      Type: Bandpass")
            print(f"      Parameters: low_cutoff={cutoff[0]} Hz, high_cutoff={cutoff[1]} Hz")
        else:
            cutoff = config.get('cutoff', 1000)
            print(f"      Type: {filter_type.capitalize()}")
            print(f"      Parameters: cutoff={cutoff} Hz")
        
        print(f"      Purpose: Remove unwanted frequency bands")
        current = apply_frequency_filter(current, sr, filter_type, cutoff)
        print(f"      ✓ Frequency filtering complete\n")

    # Step 4: STFT and Spectrogram
    if config.get('stft', True):
        print_step(4, 6, "Computing STFT and spectrogram...")
        n_fft = config.get('n_fft', 2048)
        hop_length = config.get('hop_length', 512)
        print(f"      Parameters: n_fft={n_fft}, hop_length={hop_length}")
        print(f"      Purpose: Time-frequency representation")
        freq_stft, t_stft, stft_matrix = compute_stft(current, sr, n_fft, hop_length)
        print(f"      ✓ STFT computation complete\n")

    # Step 5: Quality Metrics
    print_step(5, 6, "Calculating audio quality metrics...")
    print(f"      Comparing: Processed vs Original\n")
    metrics = evaluate_audio_quality(orig, current)
    
    print(f"      📊 QUALITY METRICS:")
    print(f"      ├─ SNR (Signal-to-Noise Ratio): {metrics['snr']:.2f} dB")
    print(f"      │  Higher is better (typically > 20 dB is good)")
    print(f"      ├─ RMS (Original): {metrics['rms_clean']:.4f}")
    print(f"      └─ RMS (Processed): {metrics['rms_processed']:.4f}\n")

    # Step 6: Save Results
    print_step(6, 6, "Saving results...")
    print_header("SAVING RESULTS")
    
    enhanced_path = os.path.join(OUTPUT_DIR, "enhanced_audio.wav")
    sf.write(enhanced_path, current, sr)
    print(f"✓ Saved enhanced audio: {enhanced_path}")
    
    save_audio_results_plot(orig, current, sr, freq_stft, t_stft, stft_matrix, metrics)
    print(f"✓ Saved comparison: {os.path.join(OUTPUT_DIR, 'audio_comparison.png')}")
    
    with open(os.path.join(OUTPUT_DIR, "audio_metrics.txt"), "w") as f:
        f.write("AUDIO QUALITY METRICS\n")
        f.write("="*50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key.upper()}: {value:.4f}\n")
    print(f"✓ Saved metrics: {os.path.join(OUTPUT_DIR, 'audio_metrics.txt')}")


def save_audio_results_plot(orig, proc, sr, freq, t, stft_matrix, metrics):
    """Save comprehensive audio comparison plot."""
    duration = len(orig) / sr
    t_axis = np.linspace(0, duration, len(orig))
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(t_axis, orig, linewidth=0.5)
    plt.title("Original Audio (Time Domain)", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(t_axis, proc, linewidth=0.5, color='orange')
    plt.title("Enhanced Audio (Time Domain)", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    if stft_matrix is not None:
        plt.subplot(2, 2, 3)
        plt.pcolormesh(t, freq, 20 * np.log10(np.abs(stft_matrix) + 1e-10), shading='gouraud', cmap='viridis')
        plt.title("Spectrogram (Time-Frequency)", fontweight='bold')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='Magnitude (dB)')
    
    plt.subplot(2, 2, 4)
    plt.axis("off")
    metrics_text = f"SNR: {metrics['snr']:.2f} dB\nRMS (Original): {metrics['rms_clean']:.4f}\nRMS (Processed): {metrics['rms_processed']:.4f}"
    plt.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "audio_comparison.png")
    plt.savefig(outpath, dpi=150)
    plt.close()


def get_user_input() -> Tuple[str, str, Dict[str, Any]]:
    """
    Gathers interactive CLI input.
    Returns (media_type, filepath, config dict)
    """
    print("\n" + "="*80)
    print("UNIVERSAL MEDIA ENHANCER".center(80))
    print("="*80)
    print("A comprehensive DSP/DIP toolkit for image and audio enhancement\n")
    
    media_choice = ""
    while media_choice not in {'i', 'a'}:
        media_choice = input("Select media type - (i)mage or (a)udio: ").strip().lower()
    
    media_type = "image" if media_choice == "i" else "audio"
    file_path = input(f"\nEnter path to {media_type} file: ").strip()

    config: Dict[str, Any] = {}
    
    if media_type == "image":
        print("\n" + "-"*80)
        print("IMAGE PROCESSING OPTIONS".center(80))
        print("-"*80 + "\n")
        
        config['gaussian'] = input("Apply Gaussian smoothing? [Y/n]: ").strip().lower() != 'n'
        if config.get('gaussian'):
            config['gaussian_kernel'] = int(input("  Gaussian kernel size (odd, e.g., 5): ") or "5")
            config['gaussian_sigma'] = float(input("  Gaussian sigma (e.g., 1.0): ") or "1.0")
        
        config['median'] = input("\nApply Median filter? [Y/n]: ").strip().lower() != 'n'
        if config.get('median'):
            config['median_kernel'] = int(input("  Median kernel size (odd, e.g., 5): ") or "5")
        
        config['fft_filter'] = input("\nApply FFT frequency filter? [y/N]: ").strip().lower() == 'y'
        if config.get('fft_filter'):
            config['fft_type'] = input("  Filter type (lowpass/highpass) [lowpass]: ") or "lowpass"
            config['fft_cutoff'] = float(input("  Cutoff frequency (0-0.5) [0.1]: ") or "0.1")
        
        config['edge_detection'] = input("\nPerform edge detection? [y/N]: ").strip().lower() == 'y'
        if config.get('edge_detection'):
            config['canny_low'] = int(input("  Canny low threshold [50]: ") or "50")
            config['canny_high'] = int(input("  Canny high threshold [150]: ") or "150")
        
        config['hist_eq'] = input("\nApply histogram equalization? [y/N]: ").strip().lower() == 'y'
        if config.get('hist_eq'):
            config['hist_method'] = input("  Method (global/clahe) [clahe]: ") or "clahe"
        
        config['wiener'] = input("\nApply Wiener denoising? [y/N]: ").strip().lower() == 'y'
        if config.get('wiener'):
            config['wiener_kernel'] = int(input("  Wiener kernel size [5]: ") or "5")
    
    else:  # audio
        print("\n" + "-"*80)
        print("AUDIO PROCESSING OPTIONS".center(80))
        print("-"*80 + "\n")
        
        config['fft_analysis'] = input("Perform FFT analysis? [Y/n]: ").strip().lower() != 'n'
        
        config['freq_filter'] = input("\nApply frequency filtering? [y/N]: ").strip().lower() == 'y'
        if config.get('freq_filter'):
            config['filter_type'] = input("  Filter type (lowpass/highpass/bandpass) [bandpass]: ") or "bandpass"
            if config['filter_type'] == "bandpass":
                config['low_cutoff'] = int(input("    Low cutoff frequency (Hz) [300]: ") or "300")
                config['high_cutoff'] = int(input("    High cutoff frequency (Hz) [3400]: ") or "3400")
            else:
                config['cutoff'] = int(input("    Cutoff frequency (Hz) [1000]: ") or "1000")
        
        config['stft'] = input("\nCompute STFT/spectrogram? [Y/n]: ").strip().lower() != 'n'
        if config.get('stft'):
            config['n_fft'] = int(input("  FFT size [2048]: ") or "2048")
            config['hop_length'] = int(input("  Hop length [512]: ") or "512")
    
    return media_type, file_path, config


def main():
    """Main application entry point."""
    try:
        media_type, file_path, config = get_user_input()
        
        if media_type == "image":
            process_image(file_path, config)
        else:
            process_audio(file_path, config)
        
        print("\n" + "="*80)
        print("✓ PROCESSING COMPLETE!".center(80))
        print("="*80 + "\n")
        print(f"Check the '{OUTPUT_DIR}/' directory for:")
        print("  • Enhanced media file")
        print("  • Comparison visualizations")
        print("  • Quality metrics report\n")
        print("Thank you for using Universal Media Enhancer!\n")
        
    except Exception as e:
        print(f"\n[ERROR]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
