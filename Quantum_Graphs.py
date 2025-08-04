"""
Quantum Gravitational Wave Detection - Plotting Module

This module contains all plotting and visualization functions for the quantum
gravitational wave detector. Separating these functions improves code organization
and makes the main detector class cleaner as I felt it was getting to busy

Author: Andrew Washburn
Last update: July 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import Dict
import seaborn as sns
from typing import Dict, Tuple
from scipy.signal import correlate
import os


def plot_wave_comparison(detector_instance, analysis_results: Dict, max_points: int = 10000, time_limit: float = 10.0):
    """
    Plot the target wave alongside the best classical and quantum detected waves.
    
    This visualization function creates a comprehensive comparison showing:
    1. Original noisy target signal
    2. Best classical template match
    3. Best quantum template match
    4. Clean reference signal (if available)
    5. Normalized overlay for direct comparison
    
    Args:
        detector_instance: Instance of QuantumGWDetector class
        analysis_results: Results from analyze_all_templates
        max_points: Maximum number of points to plot (for performance)
        time_limit: Maximum time in seconds to display (default: 10.0 seconds)
    """
    # Load clean reference signal from wave_no_noise.h5 for comparison
    try:
        with h5py.File("CW_Data/templates/wave_no_noise.h5", 'r') as f:
            clean_times = f['times'][:]
            clean_signal = f['signal'][:]
    except FileNotFoundError:
        print("Warning: Clean reference signal not found, continuing without it")
        clean_times = np.array([])
        clean_signal = np.array([])

    # Validate input data
    if not analysis_results or 'results' not in analysis_results:
        print("No results to plot")
        return
    
    if detector_instance.original_signal is None:
        print("Original signal not available for plotting")
        return
    
    # Extract best matches from analysis results
    best_classical = analysis_results['best_classical_snr']
    best_quantum = analysis_results['best_quantum_snr']
    
    if not best_classical['template'] or not best_quantum['template']:
        print("No valid templates found for comparison")
        return
    
    # Get the best template names
    best_classical_template = best_classical['template']
    best_quantum_template = best_quantum['template']
    
    # Prepare data for plotting
    target_signal = detector_instance.original_signal
    target_times = detector_instance.original_signal_times
    
    # Get template data from stored originals
    classical_template_data = detector_instance.original_templates.get(best_classical_template)
    quantum_template_data = detector_instance.original_templates.get(best_quantum_template)
    
    if not classical_template_data or not quantum_template_data:
        print("Template data not available for plotting")
        return
    
    # Handle complex signals by taking real part
    # Gravitational wave data is typically real-valued
    if np.iscomplexobj(target_signal):
        target_signal = np.real(target_signal)
    if np.iscomplexobj(classical_template_data['signal']):
        classical_template_data['signal'] = np.real(classical_template_data['signal'])
    if np.iscomplexobj(quantum_template_data['signal']):
        quantum_template_data['signal'] = np.real(quantum_template_data['signal'])
    if len(clean_signal) > 0 and np.iscomplexobj(clean_signal):
        clean_signal = np.real(clean_signal)
    
    # FILTER DATA TO FIRST time_limit SECONDS
    
    # Filter target signal to first time_limit seconds
    target_mask = target_times <= time_limit
    target_signal_filtered = target_signal[target_mask]
    target_times_filtered = target_times[target_mask]
    
    # Filter classical template to first time_limit seconds
    classical_times = classical_template_data['times']
    classical_signal = classical_template_data['signal']
    classical_mask = classical_times <= time_limit
    classical_template_signal = classical_signal[classical_mask]
    classical_template_times = classical_times[classical_mask]
    
    # Filter quantum template to first time_limit seconds
    quantum_times = quantum_template_data['times']
    quantum_signal = quantum_template_data['signal']
    quantum_mask = quantum_times <= time_limit
    quantum_template_signal = quantum_signal[quantum_mask]
    quantum_template_times = quantum_times[quantum_mask]
    
    # Filter clean reference signal to first time_limit seconds (if available)
    clean_signal_filtered = np.array([])
    clean_times_filtered = np.array([])
    if len(clean_signal) > 0:
        clean_mask = clean_times <= time_limit
        clean_signal_filtered = clean_signal[clean_mask]
        clean_times_filtered = clean_times[clean_mask]
    
    # Downsample for plotting performance if signals are still too long after time filtering
    if len(target_signal_filtered) > max_points:
        step = len(target_signal_filtered) // max_points
        target_signal_filtered = target_signal_filtered[::step]
        target_times_filtered = target_times_filtered[::step]
    
    if len(classical_template_signal) > max_points:
        step = len(classical_template_signal) // max_points
        classical_template_signal = classical_template_signal[::step]
        classical_template_times = classical_template_times[::step]
    
    if len(quantum_template_signal) > max_points:
        step = len(quantum_template_signal) // max_points
        quantum_template_signal = quantum_template_signal[::step]
        quantum_template_times = quantum_template_times[::step]
    
    if len(clean_signal_filtered) > max_points:
        step = len(clean_signal_filtered) // max_points
        clean_signal_filtered = clean_signal_filtered[::step]
        clean_times_filtered = clean_times_filtered[::step]

    # Create comprehensive comparison plot with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'Wave Comparison: Target vs Best Classical vs Best Quantum Detections (First {time_limit} seconds)', fontsize=16)
    
    # Plot 1: Target signal (noisy observation)
    axes[0].plot(target_times_filtered, target_signal_filtered, 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('Target Signal (Wave with Noise)', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(0, time_limit)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Best classical template match
    axes[1].plot(classical_template_times, classical_template_signal, 'g-', linewidth=1, alpha=0.8)
    axes[1].set_title(f'Best Classical Detection: {best_classical_template}\n(SNR: {best_classical["snr"]:.6f})', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlim(0, time_limit)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Best quantum template match
    axes[2].plot(quantum_template_times, quantum_template_signal, 'r-', linewidth=1, alpha=0.8)
    axes[2].set_title(f'Best Quantum Detection: {best_quantum_template}\n(SNR: {best_quantum["snr"]:.6f})', fontsize=12)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlim(0, time_limit)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Normalized overlay comparison for direct visual comparison
    # Normalize all signals to same amplitude scale for better visual comparison
    target_norm = target_signal_filtered / np.max(np.abs(target_signal_filtered)) if len(target_signal_filtered) > 0 and np.max(np.abs(target_signal_filtered)) > 0 else np.array([])
    classical_norm = classical_template_signal / np.max(np.abs(classical_template_signal)) if len(classical_template_signal) > 0 and np.max(np.abs(classical_template_signal)) > 0 else np.array([])
    quantum_norm = quantum_template_signal / np.max(np.abs(quantum_template_signal)) if len(quantum_template_signal) > 0 and np.max(np.abs(quantum_template_signal)) > 0 else np.array([])
    clean_norm = clean_signal_filtered / np.max(np.abs(clean_signal_filtered)) if len(clean_signal_filtered) > 0 and np.max(np.abs(clean_signal_filtered)) > 0 else np.array([])

    # Plot all normalized signals on same axes
    if len(target_norm) > 0:
        axes[3].plot(target_times_filtered, target_norm, 'b-', linewidth=1.5, alpha=0.7, label='Target Signal (Noisy)')
    if len(clean_norm) > 0:
        axes[3].plot(clean_times_filtered, clean_norm, 'k-', linewidth=1.5, alpha=0.7, label='Original Signal (Clean)')
    if len(classical_norm) > 0:
        axes[3].plot(classical_template_times, classical_norm, 'g--', linewidth=1.5, alpha=0.7, label=f'Best Classical (SNR: {best_classical["snr"]:.4f})')
    if len(quantum_norm) > 0:
        axes[3].plot(quantum_template_times, quantum_norm, 'r:', linewidth=2, alpha=0.7, label=f'Best Quantum (SNR: {best_quantum["snr"]:.4f})')

    axes[3].set_title('Normalized Overlay Comparison', fontsize=12)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Normalized Amplitude')
    axes[3].set_xlim(0, time_limit)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed first {time_limit} seconds of data")
    print(f"Target signal: {len(target_signal_filtered)} points")
    print(f"Classical template: {len(classical_template_signal)} points") 
    print(f"Quantum template: {len(quantum_template_signal)} points")
    print(f"Clean reference: {len(clean_signal_filtered)} points")


def plot_snr_comparison(analysis_results: Dict):
    """
    Plot SNR comparison results.
    
    Args:
        analysis_results: Results from analyze_all_templates
    """
    if not analysis_results or 'results' not in analysis_results:
        print("No results to plot")
        return
    
    results = analysis_results['results']
    templates = list(results.keys())
    
    if not templates:
        print("No template results to plot")
        return
    
    # Extract metrics
    classical_snrs = [results[t]['classical_snr'] for t in templates]
    quantum_snrs = [results[t]['quantum_snr'] for t in templates]
    snr_ratios = [results[t]['snr_ratio'] for t in templates]
    quantum_advantages = [results[t]['quantum_advantage'] for t in templates]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Classical vs Quantum SNR Analysis', fontsize=16)
    
    # SNR Comparison
    x = np.arange(len(templates))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, classical_snrs, width, label='Classical SNR', alpha=0.7, color='blue')
    axes[0, 0].bar(x + width/2, quantum_snrs, width, label='Quantum SNR', alpha=0.7, color='red')
    axes[0, 0].set_title('SNR Comparison')
    axes[0, 0].set_ylabel('SNR')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([t[:10] + '...' if len(t) > 10 else t for t in templates], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SNR Ratio
    colors = ['green' if adv else 'red' for adv in quantum_advantages]
    axes[0, 1].bar(x, snr_ratios, alpha=0.7, color=colors)
    axes[0, 1].set_title('SNR Ratio (Quantum/Classical)')
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([t[:10] + '...' if len(t) > 10 else t for t in templates], rotation=45)
    axes[0, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Equal performance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: Quantum vs Classical
    axes[1, 0].scatter(classical_snrs, quantum_snrs, alpha=0.7, c=colors, s=60)
    max_snr = max(max(classical_snrs), max(quantum_snrs))
    axes[1, 0].plot([0, max_snr], [0, max_snr], 'k--', alpha=0.5, label='Equal performance')
    axes[1, 0].set_xlabel('Classical SNR')
    axes[1, 0].set_ylabel('Quantum SNR')
    axes[1, 0].set_title('Quantum vs Classical SNR')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quantum Advantage Distribution - Fixed pie chart
    advantage_count = sum(quantum_advantages)
    disadvantage_count = len(templates) - advantage_count
    
    # Only create pie chart if there are results to show
    if advantage_count > 0 or disadvantage_count > 0:
        # Create pie chart with proper parameters
        wedges, texts, autotexts = axes[1, 1].pie(
            [advantage_count, disadvantage_count], 
            labels=['Quantum Advantage', 'Classical Better'], 
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90
        )
        # Apply alpha to wedges after creation
        for wedge in wedges:
            wedge.set_alpha(0.7)
    else:
        axes[1, 1].text(0.5, 0.5, 'No data to display', 
                      horizontalalignment='center', 
                      verticalalignment='center',
                      transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_title('Quantum Advantage Distribution')
    
    plt.tight_layout()
    plt.show()


def plot_comprehensive_analysis(detector_instance, analysis_results: Dict, time_limit: float = 10.0):
    """
    Plot comprehensive analysis including both SNR comparison and wave visualization.
    
    Args:
        detector_instance: Instance of QuantumGWDetector class
        analysis_results: Results from analyze_all_templates
        time_limit: Maximum time in seconds to display for wave comparison
    """
    print(f"\nGenerating comprehensive analysis plots (first {time_limit} seconds)...")
    
    # Plot SNR comparisons
    plot_snr_comparison(analysis_results)
    
    # Plot wave comparisons with time limit
    plot_wave_comparison(detector_instance, analysis_results, time_limit=time_limit)


def print_snr_summary(detector_instance, analysis_results: Dict):
    """Print detailed SNR analysis summary."""
    print(f"\n{'='*70}")
    print("SNR ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    results = analysis_results['results']
    if not results:
        print("No results to summarize")
        return
    
    templates = list(results.keys())
    classical_snrs = [results[t]['classical_snr'] for t in templates]
    quantum_snrs = [results[t]['quantum_snr'] for t in templates]
    snr_ratios = [results[t]['snr_ratio'] for t in templates if np.isfinite(results[t]['snr_ratio'])]
    quantum_advantages = [results[t]['quantum_advantage'] for t in templates]
    
    # Best results
    best_classical = analysis_results['best_classical_snr']
    best_quantum = analysis_results['best_quantum_snr']
    
    print(f"\nBest Classical SNR:")
    print(f"  Template: {best_classical['template']}")
    print(f"  Classical SNR: {best_classical['snr']:.6f}")
    print(f"  Quantum SNR: {best_classical['metrics']['quantum_snr']:.6f}")
    print(f"  Ratio: {best_classical['metrics']['snr_ratio']:.4f}")
    print(f"\nBest Quantum SNR:")
    print(f"  Template: {best_quantum['template']}")
    print(f"  Quantum SNR: {best_quantum['snr']:.6f}")
    print(f"  Classical SNR: {best_quantum['metrics']['classical_snr']:.6f}")
    print(f"  Ratio: {best_quantum['metrics']['snr_ratio']:.4f}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Classical SNR - Mean: {np.mean(classical_snrs):.6f}, Std: {np.std(classical_snrs):.6f}")
    print(f"  Quantum SNR - Mean: {np.mean(quantum_snrs):.6f}, Std: {np.std(quantum_snrs):.6f}")
    if snr_ratios:
        print(f"  SNR Ratio - Mean: {np.mean(snr_ratios):.4f}, Std: {np.std(snr_ratios):.4f}")
    print(f"  Quantum Advantage: {sum(quantum_advantages)}/{len(templates)} templates ({100*sum(quantum_advantages)/len(templates):.1f}%)")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Qubits: {detector_instance.n_qubits}")
    print(f"  Shots: {detector_instance.shots}")
    print(f"  Vector length: {detector_instance.vector_len}")


# Additional utility plotting functions

def plot_signal_overview(detector_instance, signal_path: str, time_limit: float = None):
    """
    Plot an overview of a single signal file.
    
    Args:
        detector_instance: Instance of QuantumGWDetector class
        signal_path: Path to the signal file
        time_limit: Optional time limit for display
    """
    signal, times = detector_instance.load_signal(signal_path)
    if signal is None:
        print(f"Failed to load signal from {signal_path}")
        return
    
    # Handle complex signals
    if np.iscomplexobj(signal):
        signal = np.real(signal)
    
    # Apply time limit if specified
    if time_limit is not None:
        mask = times <= time_limit
        signal = signal[mask]
        times = times[mask]
        title_suffix = f" (First {time_limit} seconds)"
    else:
        title_suffix = ""
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, signal, 'b-', linewidth=0.8, alpha=0.8)
    plt.title(f'Signal Overview: {signal_path}{title_suffix}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    if time_limit is not None:
        plt.xlim(0, time_limit)
    plt.tight_layout()
    plt.show()
    
    print(f"Signal length: {len(signal)} points")
    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} seconds")
    print(f"Duration: {times[-1] - times[0]:.3f} seconds")




def plot_snr_heatmap(analysis_results: Dict, save_path: str = None, figsize: Tuple[int, int] = (10, 8)):
    """
    Create a 2D grid heatmap visualization of Quantum SNR values with f0 and fdot axes.
    
    Args:
        analysis_results: Results from detector.analyze_all_templates()
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    if 'results' not in analysis_results or not analysis_results['results']:
        print("No analysis results found for heatmap")
        return None
    
    results = analysis_results['results']
    
    # Extract f0 and fdot values from template names and quantum SNR values
    f0_values = []
    fdot_values = []
    quantum_snrs = []
    
    for template_name in results.keys():
        try:
            # Parse template name: template_f(f0)_fdot(fdot).h5
            parts = template_name.replace('.h5', '').split('_')
            f0_part = [p for p in parts if p.startswith('f')][0]  # Find f0 part
            fdot_part = [p for p in parts if p.startswith('fdot')][0]  # Find fdot part
            
            f0 = float(f0_part[1:])  # Remove 'f' prefix
            fdot = float(fdot_part[4:])  # Remove 'fdot' prefix
            
            f0_values.append(f0)
            fdot_values.append(fdot)
            quantum_snrs.append(results[template_name]['quantum_snr'])
            
        except (ValueError, IndexError) as e:
            print(f"Could not parse template name {template_name}: {e}")
            continue
    
    if not f0_values:
        print("No valid template names found for parsing f0 and fdot")
        return None
    
    # Convert to numpy arrays
    f0_values = np.array(f0_values)
    fdot_values = np.array(fdot_values)
    quantum_snrs = np.array(quantum_snrs)
    
    # Create unique sorted values for grid
    unique_f0 = np.unique(f0_values)
    unique_fdot = np.unique(fdot_values)
    
    # Create 2D grid for heatmap
    snr_grid = np.full((len(unique_fdot), len(unique_f0)), np.nan)
    
    # Fill the grid with SNR values
    for f0, fdot, snr in zip(f0_values, fdot_values, quantum_snrs):
        f0_idx = np.where(unique_f0 == f0)[0][0]
        fdot_idx = np.where(unique_fdot == fdot)[0][0]
        snr_grid[fdot_idx, f0_idx] = snr
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(snr_grid, cmap='plasma', aspect='auto', origin='lower',
                   extent=[unique_f0.min(), unique_f0.max(), unique_fdot.min(), unique_fdot.max()])
    
    # Set labels and title
    ax.set_xlabel('f₀ (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('f̃ (Hz/s)', fontsize=12, fontweight='bold')  # fdot with dot notation
    ax.set_title('Quantum SNR Heatmap', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Quantum SNR')
    cbar.ax.tick_params(labelsize=10)
    
    # Set tick locations for better readability
    ax.set_xticks(unique_f0[::max(1, len(unique_f0)//10)])  # Show at most 10 ticks
    ax.set_yticks(unique_fdot[::max(1, len(unique_fdot)//10)])  # Show at most 10 ticks
    
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add text annotation for best quantum match
    best_quantum = analysis_results['best_quantum_snr']
    ax.text(0.02, 0.98, 
            f'Best Quantum Match:\n{best_quantum["template"].replace(".h5", "")}\nSNR: {best_quantum["snr"]:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Quantum SNR heatmap saved to: {save_path}")
    
    plt.show()
    return fig
def plot_matched_filter_response(detector, analysis_results: Dict, template_name: str = None, 
                                save_path: str = None, figsize: Tuple[int, int] = (12, 6)):
    """
    Visualize the matched filter showing the correlation between signal and template for first 10 seconds.
    
    Args:
        detector: QuantumGWDetector instance with loaded data
        analysis_results: Results from detector.analyze_all_templates()
        template_name: Specific template to analyze (if None, uses best classical match)
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    if not hasattr(detector, 'original_signal') or detector.original_signal is None:
        print("No original signal data found in detector")
        return None
    
    # Select template to analyze
    if template_name is None:
        template_name = analysis_results['best_classical_snr']['template']
        print(f"Using best classical match: {template_name}")
    
    if template_name not in detector.original_templates:
        print(f"Template {template_name} not found in loaded templates")
        return None
    
    # Get original signal and template data
    signal = detector.original_signal
    signal_times = detector.original_signal_times
    template_data = detector.original_templates[template_name]
    template = template_data['signal']
    template_times = template_data['times']
    
    # Limit to first 10 seconds of data
    time_limit = 10.0
    
    # Find indices for first 10 seconds
    signal_mask = signal_times <= time_limit
    template_mask = template_times <= time_limit
    
    signal_10s = signal[signal_mask]
    signal_times_10s = signal_times[signal_mask]
    template_10s = template[template_mask]
    template_times_10s = template_times[template_mask]
    
    # Ensure both signals have the same length for correlation
    min_length = min(len(signal_10s), len(template_10s))
    signal_10s = signal_10s[:min_length]
    template_10s = template_10s[:min_length]
    time_axis = signal_times_10s[:min_length]
    
    # Calculate the matched filter output using cross-correlation
    # This gives us the similarity measure at different time offsets
    correlation = correlate(signal_10s, template_10s, mode='same')
    
    # Create time axis for correlation (centered around zero lag)
    correlation_time = time_axis - time_axis[len(time_axis)//2]
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot all three signals
#    ax.plot(time_axis, signal_10s, 'b-', alpha=0.7, linewidth=1.5, label='Target Signal')
    ax.plot(time_axis, template_10s, 'r-', alpha=0.7, linewidth=1.5, label='Template')
    
    # Scale correlation for better visibility and plot on same axes
    correlation_scaled = correlation / np.max(np.abs(correlation)) * np.max(np.abs(signal_10s))
    ax.plot(time_axis, correlation_scaled, 'g-', alpha=0.8, linewidth=2, 
            label='Matched Filter Output (Normalized)')

    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude (Normalized)', fontsize=12, fontweight='bold')
    ax.set_title(f'Matched Filter Analysis: {template_name.replace(".h5", "")} (First 10 seconds)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits to exactly 10 seconds
    ax.set_xlim(0, min(time_limit, time_axis[-1]))
    
    # Add statistics text box
    if template_name in analysis_results['results']:
        metrics = analysis_results['results'][template_name]
        max_correlation = np.max(np.abs(correlation))
        correlation_snr = max_correlation / np.std(correlation)
        
        stats_text = f"""Matched Filter Statistics:
Classical SNR: {metrics['classical_snr']:.4f}
Quantum SNR: {metrics['quantum_snr']:.4f}
Quantum Advantage: {metrics['snr_ratio']:.3f}x
Max Correlation: {max_correlation:.4e}
Correlation SNR: {correlation_snr:.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matched filter visualization saved to: {save_path}")
    
    plt.show()
    return fig
