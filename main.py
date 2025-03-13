import json

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import glob
from collections import defaultdict
from scipy import signal
from pathlib import Path
from icecream import ic


class ConvergenceAnalyzer:
    def __init__(self, config=None):
        """
        Initialize the convergence analyzer with optional configuration

        Parameters:
        -----------
        config : dict
            Configuration similar to cWDict structure
        """
        self.config = config or {}
        self.results = {}

    def load_data(self, file_path, time_col=0, data_col=1, skip_rows=1):
        """
        Load data from OpenFOAM output file

        Parameters:
        -----------
        file_path : str
            Path to the data file
        time_col : int
            Column index for time values (0-based)
        data_col : int
            Column index for data values (0-based)
        skip_rows : int
            Number of header rows to skip

        Returns:
        --------
        tuple : (times, values)
            Arrays of time points and corresponding values
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            if "wake" in file_path:
                # Read file with whitespace as a separator
                df = pd.read_csv(file_path, sep=r"\s+", skiprows=4, names=["time", "vector1", "vector2", "vector3"])

                # Remove parentheses from vector1 and convert all columns to float
                df["vector1"] = df["vector1"].str.replace("(", "", regex=False).astype(float)
                df["vector3"] = df["vector3"].str.replace(")", "", regex=False).astype(float)

                # Compute mean of the three vector components
                df["vector_mean"] = df[["vector1", "vector2", "vector3"]].astype(float).mean(axis=1)

                times = df["time"].to_numpy()[:-1]
                values = df["vector1"].to_numpy()[:-1]
            else:
                data = np.loadtxt(file_path, skiprows=skip_rows)
                times = data[:, time_col]
                values = data[:, data_col]
            return times, values
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {str(e)}")

    def detect_oscillation(self, times, values, sampling_freq=None, cutoff_freq=0.2):
        """
        Detect a full oscillation of the low-frequency component of the signal.

        Parameters:
        -----------
        times : array
            Time values (x-axis)
        values : array
            Signal values (y-axis)
        sampling_freq : float, optional
            Sampling frequency in Hz. If None, it will be estimated.
        cutoff_freq : float, optional
            Cutoff frequency for the low-pass filter in Hz.

        Returns:
        --------
        tuple : (start_idx, end_idx, filtered_data)
            Indices marking a full oscillation and filtered signal
        """
        # Ensure we have numpy arrays
        times = np.array(times)
        values = np.array(values)

        # Need at least 4 points for filtering
        if len(times) < 4:
            return None, None, values

        # Calculate sampling frequency if not provided
        if sampling_freq is None:
            # Estimate from average time step
            dt = np.mean(np.diff(times))
            sampling_freq = 1.0 / dt

        # Apply a low-pass filter to isolate the low-frequency component
        try:
            nyquist = 0.5 * sampling_freq
            normalized_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            filtered_data = signal.filtfilt(b, a, values)


        except Exception as e:
            print(f"Warning: Filtering failed: {e}")
            return None, None, values

        mean_idx = np.argmin(np.abs(times - (times[-1]-9.5)))

        # Find zero crossings to identify oscillations (centered around mean)
        zero_crossings = np.where(np.diff(np.signbit(filtered_data - np.mean(filtered_data[mean_idx:-1]))))[0]
        # If we don't find enough zero crossings for a full oscillation, try peak detection
        if len(zero_crossings) < 3:  # Need at least 3 crossings for a full oscillation
            return None, None, filtered_data
        else:
            # Use zero crossings to identify a full oscillation
            # For a simple sine wave, a full oscillation requires 3 zero crossings
            if len(zero_crossings) >= 3:
                # Take the last full oscillation
                start_idx = zero_crossings[-3]
                end_idx = zero_crossings[-1]

                # # Create a figure with two subplots
                # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                #
                # # Plot original and filtered data in the first subplot
                # ax1.plot(times, values, label="Original Data")
                # ax1.plot(times, filtered_data, label="Filtered Data")
                # ax1.set_xlabel("Time [s]")
                # ax1.set_ylabel("Value")
                # ax1.axhline(y=np.mean(filtered_data[mean_idx:-1]), color='r', linestyle='--', label="Mean")
                # ax1.legend()
                #
                # # Plot zoomed-in data in the second subplot
                # ax2.plot(times[start_idx:end_idx], values[start_idx:end_idx], label="Original Data")
                # ax2.plot(times[start_idx:end_idx], filtered_data[start_idx:end_idx], label="Filtered Data")
                # ax2.set_xlabel("Time [s]")
                # ax2.set_ylabel("Value")
                # ax2.axhline(y=np.mean(filtered_data[mean_idx:-1]), color='r', linestyle='--', label="Mean")
                # ax2.legend()
                #
                # # Show the figure with subplots
                # plt.tight_layout()
                # plt.show()



            else:
                # Not enough zero crossings
                return None, None, filtered_data

        return start_idx, end_idx, filtered_data

    def calculate_oscillation_statistics(self, times, values, sampling_freq=None, cutoff_freq=0.2):
        """
        Calculate statistics based on the last full oscillation.

        Parameters:
        -----------
        times : array
            Time values (x-axis)
        values : array
            Signal values (y-axis)
        sampling_freq : float, optional
            Sampling frequency in Hz. If None, it will be estimated.
        cutoff_freq : float, optional
            Cutoff frequency for low-pass filtering.

        Returns:
        --------
        dict : Statistics including mean, std, slope, oscillation boundaries
        """
        # Detect oscillation
        start_idx, end_idx, filtered_data = self.detect_oscillation(times, values, sampling_freq, cutoff_freq)

        # Initialize statistics dictionary
        stats = {}

        if start_idx is None or end_idx is None:
            # No clear oscillation detected, use fallback (return None to indicate fallback needed)
            return None

        # Calculate statistics over the detected oscillation
        oscillation_times = times[start_idx:end_idx + 1]
        oscillation_values = values[start_idx:end_idx + 1]

        # Mean using trapezoidal integration for better accuracy
        try:
            from scipy import integrate
            mean = integrate.trapezoid(oscillation_values, x=oscillation_times) / (oscillation_times[-1] - oscillation_times[0])
        except ImportError:
            # Fallback to numpy if scipy.integrate is not available
            mean = np.trapz(oscillation_values, x=oscillation_times) / (oscillation_times[-1] - oscillation_times[0])

        # Standard deviation over the oscillation
        std = np.std(oscillation_values, ddof=1)

        # Calculate slope over the entire dataset for trend analysis
        lineregress = np.polyfit(times, values, 1)
        slope = lineregress[0]
        diff_height = slope * (times[-1] - times[0])

        # Store filtered data along with the times that correspond to it
        filtered_times = times

        # Assemble statistics
        stats = {
            'mean': mean,
            'std': std,
            'slope': slope,
            'diff_height': diff_height,
            'begin': oscillation_times[0],
            'end': oscillation_times[-1],
            'cut_times': oscillation_times,
            'cut_values': oscillation_values,
            'filtered_times': filtered_times,
            'filtered_data': filtered_data,
            'used_oscillation_detection': True,
            'cutoff_freq': cutoff_freq
        }

        return stats

    def statistic(self, series_one, series_two, start, end, detect_oscillations=False, cutoff_freq=0.2):
        """
        Calculate statistics on a time window, optionally using oscillation detection
        Based on the statistic function in cW.py

        Parameters:
        -----------
        series_one : array
            Time points
        series_two : array
            Data values
        start : float
            Start time of window
        end : float
            End time of window
        detect_oscillations : bool
            Whether to use oscillation detection
        cutoff_freq : float
            Cutoff frequency for low-pass filtering

        Returns:
        --------
        dict : Statistics including mean, std, slope, diff_height
        """
        # Try oscillation detection if requested
        if detect_oscillations:
            osc_stats = self.calculate_oscillation_statistics(series_one, series_two, cutoff_freq=cutoff_freq)
            if osc_stats is not None:
                return osc_stats

            # If oscillation detection failed, fall back to standard method
            print("  Oscillation detection failed, falling back to standard method")

        # Standard method (original statistic calculation)
        # Cut series to specified window
        mask = (series_one >= start) & (series_one <= end)
        if not np.any(mask):
            return None

        cut_series_one = series_one[mask]
        cut_series_two = series_two[mask]

        # Calculate statistics
        interval = end - start
        mean = np.trapz(cut_series_two, x=cut_series_one) / interval
        std = np.std(cut_series_two, ddof=1)

        # Calculate regression line
        try:
            coeffs = np.polyfit(cut_series_one, cut_series_two, 1)
            slope = coeffs[0]
            diff_height = slope * interval
        except:
            slope = 0
            diff_height = 0

        return {
            'mean': mean,
            'std': std,
            'slope': slope,
            'diff_height': diff_height,
            'begin': start,
            'end': end,
            'cut_times': cut_series_one,
            'cut_values': cut_series_two,
            'used_oscillation_detection': False
        }

    def check_convergence(self, quantity, times, values, start_time,
                          limit_std_dev, limit_diff_he, base_interval, relative=True,
                          detect_oscillations=False, cutoff_freq=0.2, attempt_number=0,
                          use_slope_criteria=True):
        """
        Check if a quantity meets convergence criteria with progressive interval expansion

        Parameters:
        -----------
        quantity : str
            Name of the quantity being analyzed
        times : array
            Time points
        values : array
            Data values
        start_time : float
            Minimum time to start checking convergence
        limit_std_dev : float
            Standard deviation limit for convergence
        limit_diff_he : float
            Slope/height difference limit for convergence
        base_interval : float
            Base time interval to use for analysis
        relative : bool
            Whether to use relative criteria (percentage of mean)
        detect_oscillations : bool
            Whether to use oscillation detection
        cutoff_freq : float
            Cutoff frequency for low-pass filtering
        attempt_number : int
            Current attempt number (determines interval multiplier)
        use_slope_criteria : bool
            Whether to include slope criteria in convergence check (default: True)

        Returns:
        --------
        tuple : (converged, stats, oscillation_detected)
            Boolean indicating convergence, statistics dictionary, and boolean indicating if oscillation was detected
        """
        # Skip if not enough data or haven't reached start time
        if len(times) < 10 or times[-1] < start_time:
            return False, None, False

        # Calculate the current interval based on attempt number
        # First attempt uses base_interval, each subsequent attempt adds base_interval
        current_interval = base_interval * (attempt_number + 1)

        # Define analysis window
        end = times[-1]
        start = start_time

        # Skip if window is too small
        if end - start < base_interval / 2:
            return False, None, False

        # Try oscillation detection if requested
        oscillation_detected = False
        if detect_oscillations:
            try:
                osc_stats = self.calculate_oscillation_statistics(times, values, cutoff_freq=cutoff_freq)
                if osc_stats is not None:
                    stats = osc_stats
                    oscillation_detected = True
                else:
                    # Not enough info to detect oscillation
                    return False, None, False
            except Exception as e:
                # Error in oscillation detection
                return False, None, False
        else:
            # Standard method - always use current window size
            stats = self.statistic(times, values, start, end, detect_oscillations=False)

        if stats is None:
            return False, None, oscillation_detected

        # Store results for this quantity
        self.results[quantity] = stats

        # Check convergence criteria
        std = stats['std']
        diff_height = stats['diff_height']

        if relative:
            # Convert to percentage of mean (like in cW.py)
            std = abs((std / stats['mean']) * 100)
            diff_height = abs((diff_height / stats['mean']) * 100)

        std_converged = std <= limit_std_dev
        diff_converged = True  # Default to True if not using slope criteria

        if use_slope_criteria:
            diff_converged = abs(diff_height) <= limit_diff_he

        # Both criteria must be met for convergence (if slope criteria is enabled)
        converged = std_converged and diff_converged

        # Add convergence info to stats
        stats['std_relative'] = std
        stats['diff_height_relative'] = diff_height
        stats['std_converged'] = std_converged
        stats['diff_converged'] = diff_converged
        stats['converged'] = converged
        stats['limit_std_dev'] = limit_std_dev
        stats['limit_diff_he'] = limit_diff_he
        stats['interval_used'] = current_interval
        stats['attempt_number'] = attempt_number
        stats['use_slope_criteria'] = use_slope_criteria

        return converged, stats, oscillation_detected

    def load_original_phase_summary(self, log_file="watch.log"):
        """
        Load and process the watch.log file to extract phase summary information.

        Parameters:
        -----------
        log_file : str
            Path to the watch.log file

        Returns:
        --------
        dict : Dictionary organized by quantity and phase, containing phase information
        """
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Watch log file not found: {log_file}")

        # Initialize a nested dictionary to store the data
        phase_summary = {}

        # Read the log file
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # We expect the data on the second line (after header)
        if len(lines) < 2:
            raise ValueError("Watch log file is empty or contains only header")

        # Parse data line
        data_line = lines[1].strip()
        tokens = data_line.split()

        # Define tasks to look for
        tasks = ["velocityRamp", "largeTimeStep",
                 "propulsionVariation", "final"]

        cumulative_time = 0.0
        phase_summary = defaultdict(dict)

        for task in tasks:
            task_line = tokens[tokens.index(task):]

            task_time = float(task_line[1])
            task_converged = task_line[3] == "CON"

            # Calculate task start and end times
            task_start = cumulative_time
            task_end = cumulative_time + task_time
            cumulative_time = task_end

            # Add task information to phase summary for each quantity
            phase_summary[task] = {
                'start_time': task_start,
                'end_time': task_end,
                'start_value': None,  # We don't have this info in log
                'end_value': None,  # We don't have this info in log
                'converged': task_converged,
                'convergence_time': task_end if task_converged else None,
                'convergence_value': None,  # We don't have this info in log
                'time_saved': 0.0  # Will be calculated later if needed
            }

        return phase_summary

    def plot_convergence(self, data_dict, actual_task_ranges, quantities=None, save_plots=False, output_dir=None):
        """
        Plot the convergence of selected quantities over time with phase boundaries.
        Includes a subfigure that zooms in on the final phase.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to (times, values) tuples
        actual_task_ranges : dict
            Dictionary of phase start and end times
        quantities : list or None
            List of quantities to plot, if None plots Fx and WFT if available
        save_plots : bool
            Whether to save the plots to files
        output_dir : str
            Directory to save plots to, if None uses current directory
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path

        # Default to Fx and WFT if available
        if quantities is None:
            quantities = []
            if 'Fx' in data_dict:
                quantities.append('Fx')
            if 'WFT' in data_dict:
                quantities.append('WFT')

        # Ensure quantities exist in data_dict
        quantities = [q for q in quantities if q in data_dict]
        if not quantities:
            print("No valid quantities to plot")
            return

        # Prepare colors for different phases
        phase_colors = {
            'velocityRamp': 'tab:blue',
            'largeTimeStep': 'tab:green',
            'propulsionVariation': 'tab:purple',
            'final': 'tab:orange'
        }

        # Define line styles for different boundary types
        boundary_styles = {
            'start': ':',  # dotted
            'end': '--',  # dashed
            'end_original': '-.'  # dash-dot
        }

        # Create a figure for each quantity
        for quantity in quantities:
            # Create a figure with two subplots (main plot and zoomed view)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
            times, values = data_dict[quantity]

            # Plot the main data on the first subplot
            ax1.plot(times, values, 'tab:red', label=quantity, linewidth=2)

            # Collect all boundary times for setting x-axis limits
            all_boundaries = []

            # Add vertical lines for phase boundaries on the main plot
            for phase, ranges in actual_task_ranges.items():
                for boundary_type in ['start', 'end', 'end_original']:
                    if boundary_type in ranges:
                        # Convert numpy.float64 to regular float if needed
                        boundary_value = float(ranges[boundary_type])
                        all_boundaries.append(boundary_value)

                        ax1.axvline(
                            x=boundary_value,
                            color=phase_colors.get(phase, 'gray'),
                            linestyle=boundary_styles[boundary_type],
                            label=f"{phase} {boundary_type.replace('_', ' ')}",
                            alpha=0.7
                        )

            # Calculate reasonable x-axis limits based on phase boundaries for main plot
            if all_boundaries:
                x_min = max(0, min(all_boundaries) - 10)  # Start a bit before first boundary
                x_max = max(all_boundaries) + 10  # End a bit after last boundary
                ax1.set_xlim(x_min, x_max)

                # Find suitable y-axis limits for main plot
                mask = (times >= x_min) & (times <= x_max)
                if np.any(mask):
                    masked_values = values[mask]
                    y_margin = (np.max(masked_values) - np.min(masked_values)) * 0.1
                    y_min = np.min(masked_values) - y_margin
                    y_max = np.max(masked_values) + y_margin
                    ax1.set_ylim(y_min, y_max)

            # Add labels and legend to main plot
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel(f'{quantity} Value', fontsize=12)
            ax1.set_title(f'{quantity} Convergence Over Time', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Add a legend but handle potential duplicates for main plot
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)

            # Create zoomed view on the second subplot - focusing on the final phase
            if 'final' in actual_task_ranges:
                # Get the final phase boundaries
                final_start = actual_task_ranges['final']['start']
                final_end = actual_task_ranges['final']['end']

                # If there's an original end time, include it in the zoomed view
                if 'end_original' in actual_task_ranges['final']:
                    final_end_original = actual_task_ranges['final']['end_original']
                    zoom_end = max(final_end, final_end_original) + 5  # Add a bit of padding
                else:
                    zoom_end = final_end + 5

                zoom_start = final_start - 5  # Start a bit before the final phase

                # Plot the zoomed data
                ax2.plot(times, values, 'tab:red', linewidth=2)

                # Add vertical lines for final phase boundaries in zoomed view
                for boundary_type in ['start', 'end', 'end_original']:
                    if boundary_type in actual_task_ranges['final']:
                        boundary_value = float(actual_task_ranges['final'][boundary_type])
                        ax2.axvline(
                            x=boundary_value,
                            color=phase_colors.get('final', 'tab:orange'),
                            linestyle=boundary_styles[boundary_type],
                            label=f"final {boundary_type.replace('_', ' ')}",
                            alpha=0.7
                        )

                # Set the zoom range
                ax2.set_xlim(zoom_start, zoom_end)

                # Get the y-axis range for the zoomed area
                zoom_mask = (times >= zoom_start) & (times <= zoom_end)
                if np.any(zoom_mask):
                    zoom_values = values[zoom_mask]
                    y_margin = (np.max(zoom_values) - np.min(zoom_values)) * 0.1
                    zoom_y_min = np.min(zoom_values) - y_margin
                    zoom_y_max = np.max(zoom_values) + y_margin
                    ax2.set_ylim(zoom_y_min, zoom_y_max)

                # Add labels and styling to zoomed view
                ax2.set_xlabel('Time (s)', fontsize=12)
                ax2.set_ylabel(f'{quantity} Value', fontsize=12)
                ax2.set_title(f'Zoomed View of Final Phase', fontsize=14)
                ax2.grid(True, alpha=0.3)

                # Add legend to zoomed view
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)

                # Add a box on the main plot showing the zoomed area
                from matplotlib.patches import Rectangle
                zoom_rect = Rectangle((zoom_start, zoom_y_min), zoom_end - zoom_start, zoom_y_max - zoom_y_min,
                                      fill=False, edgecolor='black', linestyle=':', linewidth=1.5)
                ax1.add_patch(zoom_rect)

            plt.tight_layout()

            # Save plot if requested
            if save_plots:
                if output_dir is None:
                    output_dir = '.'
                output_path = Path(output_dir) / f'{quantity}_convergence_with_zoom.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {output_path}")

            plt.show()

    def analyze_convergence_multi_quantity(self, data_dict, tasks, required_convergence=None, phase_summary_file="watch.log",
                                           original_results="CWBot.results", plot_results=True, save_plots=False, output_dir=None):
        """
        Analyze convergence across multiple quantities with properly implemented progressive interval

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to (times, values) tuples
        tasks : list
            List of task dictionaries with configuration settings
        required_convergence : dict, optional
            Dictionary mapping task names to required quantities for convergence
        phase_summary_file : str
            Path to the original phase summary CSV file
        plot_results : bool
            Whether to plot the convergence results
        save_plots : bool
            Whether to save the plots to files
        output_dir : str
            Directory to save plots to, if None uses current directory

        Returns:
        --------
        tuple : (all_transitions, actual_task_ranges, time_saving)
            Transitions for all quantities, actual task time ranges, and time saved
        """

        # Load original phase summary from CSV
        try:
            original_phase_summary = self.load_original_phase_summary(phase_summary_file)
        except FileNotFoundError:
            print(f"Warning: Could not find phase summary file: {phase_summary_file}")
            original_phase_summary = {}  # Empty dict as fallback

        # load the original values from CWBot.results
        # Load the file
        with open(original_results, "r") as file:
            data = json.load(file)

        # Extract the tasks into a structured dictionary
        tasks_dict = {}

        for task in data["tasks"]:
            task_name = task["name"]

            # Extract actions
            actions_completed = [action["completed"] for action in task["actions"]]

            # Extract checks
            checks = {}
            for check in task["checks"]:
                check_name = check["name"]
                checks[check_name] = {
                    "mean": check["mean"],
                    "slope": check["slope"],
                    "std": check["std"],
                    "criteria": [
                        {
                            "name": crit["name"],
                            "value": crit["value"],
                            "endTime": crit["endTime"],
                            "completed": crit["completed"],
                        }
                        for crit in check.get("criteria", [])
                    ]
                }

            # Store task data
            tasks_dict[task_name] = {
                "startTime": task["startTime"],
                "endTime": task["endTime"],
                "actions_completed": actions_completed,
                "checks": checks
            }

        if required_convergence is None:
            # Default: each task requires convergence of all quantities analyzed
            required_convergence = {}

        # Dictionary to track convergence status for each task and quantity
        task_convergence = defaultdict(dict)
        all_transitions = defaultdict(list)
        current_time = 0.0

        # Track quantities that have converged for each task
        converged_quantities = defaultdict(set)

        # Track oscillation detection attempts and last check times
        oscillation_attempts = defaultdict(lambda: defaultdict(int))
        last_check_times = defaultdict(lambda: defaultdict(float))

        # For each task, we'll store the actual data range
        actual_task_ranges = {}

        # Process each task in sequence
        for task_index, task in enumerate(tasks):
            task_name = task.get('name', 'unknown')
            max_time = task.get('max_time', 0.0)

            print(f"\n=== Analyzing task: {task_name} ===")

            if task_name == "velocityRamp":
                current_time = 0.0

            if task_name == "largeTimeStep":
                current_time = original_phase_summary["velocityRamp"]["end_time"]

            # Set current time based on original phase summary if available
            if task_name == "propulsionVariation" and original_phase_summary:
                # Check if we have the necessary data in original_phase_summary
                if 'largeTimeStep' in original_phase_summary:
                    current_time = original_phase_summary["largeTimeStep"]["end_time"]

            if task_name == "final" and original_phase_summary:
                # Check if we have the necessary data in original_phase_summary
                if 'propulsionVariation' in original_phase_summary:
                    current_time = original_phase_summary["propulsionVariation"]["end_time"]

            print(f"  Start time: {current_time}")

            # Record task start for each quantity
            for quantity in data_dict.keys():
                all_transitions[quantity].append({
                    "time": current_time,
                    "type": "start",
                    "task": task_name
                })
                # Initialize last check time to start time
                last_check_times[task_name][quantity] = current_time

            # Default end time
            if task_name == "velocityRamp":
                end_time = original_phase_summary["velocityRamp"]["end_time"]
            else:
                end_time = current_time + max_time if max_time else current_time

            # Store the initial task range
            actual_task_ranges[task_name] = {"start": current_time, "end": end_time}

            # If we need to check convergence
            if all(k in task for k in ['std_limit', 'slope_limit', 'interval']):
                # Get the required quantities for this task
                required_quantities = required_convergence.get(task_name, list(data_dict.keys()))
                print(f"  Required quantities for convergence: {required_quantities}")

                # Flag to track if any quantity converged
                task_converged = False
                convergence_time = None

                # Check if slope criteria should be used for this task
                use_slope_criteria = task.get('use_slope_criteria', True)  # Default to True for backward compatibility

                if not use_slope_criteria:
                    print(f"  Note: Slope criteria is disabled for task {task_name}")

                # Check each quantity for convergence
                for quantity, (times, values) in data_dict.items():
                    # Skip if we're not tracking this quantity for the current task
                    if quantity not in required_quantities:
                        print(f"  Skipping {quantity} (not required for {task_name})")
                        continue

                    # Check if we should use oscillation detection for this quantity
                    use_oscillation = task.get('use_oscillation_detection', False) and quantity in task.get('oscillation_quantities', [])
                    cutoff_freq = task.get('oscillation_cutoff_freq', 0.2)

                    detection_method = "oscillation detection" if use_oscillation else "fixed interval"
                    print(f"  Checking {quantity} using {detection_method}:")
                    print(f"    - Standard deviation limit: {task['std_limit']}%")

                    if use_slope_criteria:
                        print(f"    - Slope limit: {task['slope_limit']}%")
                    else:
                        print(f"    - Slope criteria disabled")

                    print(f"    - Base interval: {task['interval']} s")

                    if use_oscillation:
                        print(f"    - Oscillation cutoff frequency: {cutoff_freq} Hz")

                    # Get data points in the task time range
                    task_mask = (times >= current_time) & (times <= end_time)
                    if not np.any(task_mask):
                        print(f"    No data points for {quantity} in this time range")
                        continue

                    task_times = times[task_mask]
                    task_values = values[task_mask]

                    # Initialize convergence flag
                    quantity_converged = False

                    # Get current attempt count and base interval
                    attempt = oscillation_attempts[task_name][quantity]
                    base_interval = task['interval']

                    # Calculate the current interval size based on attempt
                    current_interval = base_interval * (attempt + 1)

                    # Main convergence check loop - go through data
                    last_time = 0
                    for i in range(len(task_times)):
                        check_time = task_times[i]

                        # Skip data points we've already processed
                        if check_time <= last_check_times[task_name][quantity]:
                            continue

                        # Only check convergence when we have enough data since last check
                        # This is key - we need at least one full interval since last check
                        if check_time < last_check_times[task_name][quantity] + current_interval:
                            continue

                        last_time = check_time

                        # Log attempt info
                        print(f"    Checking convergence at t={check_time:.2f}s with interval={current_interval:.2f}s (attempt #{attempt + 1})")
                        attempt += 1

                        # Run convergence check
                        converged, stats, oscillation_detected = self.check_convergence(
                            quantity,
                            task_times[:i + 1],  # All data up to this point
                            task_values[:i + 1],
                            current_time,  # Start time remains the task start
                            task['std_limit'],
                            task['slope_limit'],
                            base_interval,  # Pass the base interval
                            relative=True,
                            detect_oscillations=use_oscillation,
                            cutoff_freq=cutoff_freq,
                            attempt_number=attempt,
                            use_slope_criteria=use_slope_criteria  # Pass the slope criteria flag
                        )

                        # Update last check time
                        last_check_times[task_name][quantity] = check_time

                        # Handle oscillation detection outcome
                        if use_oscillation:
                            if oscillation_detected:
                                # Success! Check convergence criteria
                                if converged:
                                    # Quantity has converged
                                    quantity_converged = True
                                    detection_info = "oscillation detection"
                                    interval_used = stats.get('interval_used', current_interval)

                                    print(f"    {quantity} converged at: {check_time} using {detection_info} (interval: {interval_used:.2f}s)")
                                    print(f"    Stats: mean={stats['mean']:.2f}, std={stats['std_relative']:.4f}%", end="")
                                    if use_slope_criteria:
                                        print(f", slope={stats['diff_height_relative']:.4f}%")
                                    else:
                                        print(" (slope criteria disabled)")

                                    # Record convergence
                                    all_transitions[quantity].append({
                                        "time": check_time,
                                        "type": "converge",
                                        "task": task_name,
                                        "stats": stats,
                                        "detection_method": detection_info
                                    })
                                    # Mark as converged
                                    converged_quantities[task_name].add(quantity)
                                    task_convergence[task_name][quantity] = check_time
                                    break
                                else:
                                    print(f"    Oscillation detected but convergence criteria not met, continuing to next time point")
                            else:
                                # Failed to detect oscillation with current interval size
                                print(f"    Failed to detect oscillation with interval={current_interval:.2f}s, incrementing attempt counter")
                                oscillation_attempts[task_name][quantity] += 1
                        else:
                            # Standard fixed-interval method
                            if converged:
                                quantity_converged = True
                                detection_info = "fixed interval"

                                print(f"    {quantity} converged at: {check_time} using {detection_info}")
                                print(f"    Stats: mean={stats['mean']:.2f}, std={stats['std_relative']:.4f}%", end="")
                                if use_slope_criteria:
                                    print(f", slope={stats['diff_height_relative']:.4f}%")
                                else:
                                    print(" (slope criteria disabled)")

                                # Record convergence
                                all_transitions[quantity].append({
                                    "time": check_time,
                                    "type": "converge",
                                    "task": task_name,
                                    "stats": stats,
                                    "detection_method": detection_info
                                })

                                # Mark as converged
                                converged_quantities[task_name].add(quantity)
                                task_convergence[task_name][quantity] = check_time
                                break

                    # End of time loop for this quantity
                    if not quantity_converged:
                        print(f"    {quantity} did not converge during {task_name}")

                    # Reset attempt counter for next task
                    if task_index < len(tasks) - 1:
                        next_task = tasks[task_index + 1]['name']
                        oscillation_attempts[next_task][quantity] = 0
                        last_check_times[next_task][quantity] = 0

                # Determine if the task is completely converged
                # For largeTimeStep, only check Fx
                if task_name == "largeTimeStep":
                    task_converged = "Fx" in converged_quantities[task_name]
                    if task_converged:
                        convergence_time = task_convergence[task_name]["Fx"]
                        print(f"  Task {task_name} converged based on Fx at {convergence_time}")
                    else:
                        print(f"  Task {task_name} did not converge (Fx not converged)")

                # For propulsionVariation/final, check if all required quantities converged
                elif task_name in ["propulsionVariation", "final"]:
                    # Implement AND logic - all required quantities must converge
                    if all(qty in converged_quantities[task_name] for qty in required_quantities):
                        task_converged = True
                        # Use the latest convergence time
                        convergence_time = max(task_convergence[task_name].values())
                        print(f"  Task {task_name} converged (all required quantities converged) at {convergence_time}")
                    else:
                        # Print which quantities are still waiting to converge
                        pending = [qty for qty in required_quantities if qty not in converged_quantities[task_name]]
                        print(f"  Task {task_name} waiting for convergence of: {', '.join(pending)}")
                        print(f"  Task {task_name} did not converge (not all required quantities converged)")

                # For other tasks, all required quantities must converge
                else:
                    # Default AND logic - all required quantities must converge
                    if all(qty in converged_quantities[task_name] for qty in required_quantities):
                        task_converged = True
                        # Use the latest convergence time
                        convergence_time = max(task_convergence[task_name].values())
                        print(f"  Task {task_name} converged at {convergence_time}")
                    else:
                        print(f"  Task {task_name} did not converge (not all required quantities converged)")

                # If task converged, immediately transition to next task
                if task_converged:
                    if original_phase_summary['largeTimeStep']['end_time'] > convergence_time:
                        print(f"  Task {task_name} converged early at {convergence_time}, skipping ahead to next task")
                    else:
                        print(f"  Task {task_name} converged at {convergence_time}, original end time: {original_phase_summary[f'{task_name}']['end_time']} was better")

                    # Update the actual end time for this task
                    actual_task_ranges[task_name]["end"] = round(convergence_time, 2)
                    actual_task_ranges[task_name]["end_original"] = original_phase_summary[task_name]["end_time"]

                    # Record task end for each quantity at convergence time
                    for quantity in data_dict.keys():
                        converged = quantity in converged_quantities[task_name]
                        all_transitions[quantity].append({
                            "time": convergence_time,
                            "type": "end",
                            "task": task_name,
                            "reason": "converged" if converged else "task_converged"
                        })

                    # Update current time to convergence time for next task
                    current_time = convergence_time
                else:
                    # Task didn't converge, continue to max time
                    current_time = end_time

                    # Record task end for each quantity at max time
                    for quantity in data_dict.keys():
                        converged = quantity in converged_quantities[task_name]
                        all_transitions[quantity].append({
                            "time": end_time,
                            "type": "end",
                            "task": task_name,
                            "reason": "converged" if converged else "max_time"
                        })
            else:
                # For tasks without convergence checking, just update the time
                current_time = end_time

                # Record task end for each quantity
                for quantity in data_dict.keys():
                    all_transitions[quantity].append({
                        "time": end_time,
                        "type": "end",
                        "task": task_name,
                        "reason": "completed"
                    })

        # Time saving calculation
        # Get all phases in order
        phases = list(actual_task_ranges.keys())

        # Calculate time saving between consecutive phases
        time_saving, percent_saving = {}, {}
        for phase in phases:
            if 'end_original' in actual_task_ranges[phase]:
                time_saving[phase] = actual_task_ranges[phase]['end_original'] - actual_task_ranges[phase]['end']
                percent_saving[phase] = time_saving[phase] / actual_task_ranges[phase]['end_original'] * 100

        value_difference = defaultdict(dict)
        for quantity in data_dict.keys():
            for phase in phases:
                times = data_dict[quantity][0]
                values = data_dict[quantity][1]

                end_time = actual_task_ranges[phase]["end"]
                original_end_time = original_phase_summary[phase]["end_time"]

                end_idx = np.argmin(np.abs(times - end_time))
                original_end_idx = np.argmin(np.abs(times - original_end_time))

                end_value = values[end_idx]
                original_end_value = values[original_end_idx]

                # check value wrt cWBot.results
                if quantity in tasks_dict[phase]["checks"]:
                    check = tasks_dict[phase]["checks"][quantity]
                    check_value = check["mean"]

                value_difference[quantity][phase] = (end_value - original_end_value) / original_end_value * 100

        # Plot results if requested
        if plot_results:
            self.plot_convergence(data_dict, actual_task_ranges, quantities=['Fx', 'WFT'],
                                  save_plots=save_plots, output_dir=output_dir)

        return all_transitions, actual_task_ranges, time_saving, value_difference


class MultiSimulationAnalyzer:
    """
    Class to handle multiple simulation analyses
    """

    def __init__(self, root_dir="./copied_foam_results/489_MSC/131_ECO-Retrofit_WEC-VAN-RUYSDAEL"):
        """
        Initialize the multi-simulation analyzer

        Parameters:
        -----------
        root_dir : str
            Root directory containing simulation folders
        """
        self.root_dir = root_dir
        self.simulation_dirs = []
        self.results = {}

    def find_simulation_folders(self):
        """
        Find all simulation folders in the root directory

        Returns:
        --------
        list : Paths to simulation folders
        """
        print(f"Searching for simulation folders in: {self.root_dir}")

        # Use Path for better cross-platform compatibility
        root = Path(self.root_dir)

        if not root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # Find all folders that contain both sixDoF and postProcessing
        sim_folders = []

        for folder in root.iterdir():
            if not folder.is_dir():
                continue

            # Check if this folder contains the required subfolders and CSV file
            has_sixdof = (folder / "sixDoF").exists()
            has_postprocessing = (folder / "postProcessing").exists()
            has_csv = any((folder / f).exists() for f in ["convergence_summary.csv", "original_phase_summary.csv"])

            if has_sixdof and has_postprocessing and has_csv:
                sim_folders.append(folder)
                print(f"Found simulation folder: {folder}")

        if not sim_folders:
            print(f"Warning: No simulation folders found in {self.root_dir}")
        else:
            print(f"Found {len(sim_folders)} simulation folders")

        self.simulation_dirs = sim_folders
        return sim_folders

    def process_simulation(self, sim_folder, tasks, required_convergence=None, plot_results=True, save_plots=False, output_dir=None):
        """
        Process a single simulation folder

        Parameters:
        -----------
        sim_folder : str or Path
            Path to simulation folder
        tasks : list
            List of task dictionaries with configuration settings
        required_convergence : dict, optional
            Dictionary mapping task names to required quantities for convergence
        plot_results : bool
            Whether to plot the convergence results
        save_plots : bool
            Whether to save the plots to files
        output_dir : str
            Directory to save plots to, if None uses current directory

        Returns:
        --------
        dict : Results for this simulation
        """
        sim_folder = Path(sim_folder) if isinstance(sim_folder, str) else sim_folder
        sim_name = sim_folder.name

        print(f"\n{'=' * 80}")
        print(f"Processing simulation: {sim_name}")
        print(f"{'=' * 80}")

        # Initialize analyzer for this simulation
        analyzer = ConvergenceAnalyzer()

        # Find CSV file (either convergence_summary.csv or original_phase_summary.csv)
        watchlog_file = None
        if (sim_folder / "watch.log").exists():
            watchlog_file = str(sim_folder / "watch.log")

        if (sim_folder / "CWBot.results").exists():
            cwbot_file = str(sim_folder / "CWBot.results")

        if not watchlog_file:
            print(f"Warning: No summary CSV file found in {sim_folder}")
            return None

        print(f"Using summary file: {watchlog_file}")

        # Set output directory for this simulation if saving plots
        if save_plots and output_dir is None:
            sim_output_dir = sim_folder / "plots"
            os.makedirs(sim_output_dir, exist_ok=True)
        else:
            sim_output_dir = output_dir

        # Try to load data files for this simulation
        try:
            data_dict = {
                "Fx": analyzer.load_data(str(sim_folder / "sixDoF/HULL/0/forces.dat"), time_col=0, data_col=1),
                "FxP": analyzer.load_data(str(sim_folder / "sixDoF/HULL/0/forcesPV.dat"), time_col=0, data_col=1),
                "FxV": analyzer.load_data(str(sim_folder / "sixDoF/HULL/0/forcesPV.dat"), time_col=0, data_col=7),
                "Trim": analyzer.load_data(str(sim_folder / "sixDoF/HULL/0/position.dat"), time_col=0, data_col=5),
                "Sinkage": analyzer.load_data(str(sim_folder / "postProcessing/sinkage/0/relativeMotion.dat"), time_col=0, data_col=3),
                "WFT": analyzer.load_data(str(sim_folder / "postProcessing/wake/0/volFieldValue.dat"), time_col=0, data_col=1),
                "Wetted": analyzer.load_data(str(sim_folder / "postProcessing/wettedSurface/0/surfaceFieldValue.dat"), time_col=0, data_col=1)
            }

            # Filter out any data that couldn't be loaded
            valid_data_dict = {k: v for k, v in data_dict.items() if v is not None}

            if not valid_data_dict:
                print(f"Warning: No valid data files found in {sim_folder}")
                return None

            print(f"Loaded data for quantities: {', '.join(valid_data_dict.keys())}")

            # Analyze convergence
            all_transitions, actual_task_ranges, time_saving, value_difference = analyzer.analyze_convergence_multi_quantity(
                valid_data_dict,
                tasks,
                required_convergence,
                phase_summary_file=watchlog_file,
                original_results=cwbot_file,
                plot_results=plot_results,
                save_plots=save_plots,
                output_dir=sim_output_dir
            )

            print(f"\nResults for simulation {sim_name}:")

            # Get the total simulation time (end of final task)
            if 'final' in actual_task_ranges:
                total_time = actual_task_ranges['final']['end']

            # Store results for this simulation
            sim_results = {
                "name": sim_name,
                "transitions": all_transitions,
                "task_ranges": actual_task_ranges,
                "time_saving": sum(time_saving.values()) if time_saving else None,
                "time_saving_percentage": sum(time_saving.values()) / total_time * 100 if time_saving and total_time else None,
                "value_difference": value_difference
            }

            return sim_results

        except Exception as e:
            print(f"Error processing simulation {sim_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def process_all_simulations(self, tasks, required_convergence=None, plot_results=True, save_plots=False, output_dir=None):
        """
        Process all simulation folders

        Parameters:
        -----------
        tasks : list
            List of task dictionaries with configuration settings
        required_convergence : dict, optional
            Dictionary mapping task names to required quantities for convergence
        plot_results : bool
            Whether to plot the convergence results
        save_plots : bool
            Whether to save the plots to files
        output_dir : str
            Directory to save plots to, if None uses current directory

        Returns:
        --------
        dict : Results for all simulations
        """
        if not self.simulation_dirs:
            self.find_simulation_folders()

        if not self.simulation_dirs:
            print("No simulation folders found, aborting.")
            return {}

        # Create output directory if saving plots
        if save_plots and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Process each simulation folder
        all_results = {}

        for sim_folder in self.simulation_dirs:
            sim_results = self.process_simulation(
                sim_folder,
                tasks,
                required_convergence,
                plot_results=plot_results,
                save_plots=save_plots,
                output_dir=output_dir
            )

            # Print final differences


            if sim_results:
                all_results[sim_folder.name] = sim_results

        # Summarize results
        print("\n" + "=" * 80)
        print(f"Processed {len(all_results)} simulations successfully")
        print("=" * 80)

        return all_results

    def export_results_to_csv(self, output_file="multi_simulation_results.csv"):
        """
        Export summary results to a CSV file

        Parameters:
        -----------
        output_file : str
            Path to output CSV file
        """
        if not self.results:
            print("No results to export")
            return

        # Prepare data for CSV
        data = []

        for sim_name, sim_results in self.results.items():
            # Get task convergence times
            task_data = {}

            for task_name, task_range in sim_results["task_ranges"].items():
                task_data[f"{task_name}_start"] = task_range["start"]
                task_data[f"{task_name}_end"] = task_range["end"]
                task_data[f"{task_name}_duration"] = task_range["end"] - task_range["start"]

            # Add row for this simulation
            row = {
                "simulation": sim_name,
                "time_saving": sim_results["time_saving"],
                "time_saving_percentage": sim_results["time_saving_percentage"]
            }
            row.update(task_data)

            data.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        print(f"Results exported to {output_file}")


if __name__ == "__main__":
    # Example configuration with oscillation detection and optional slope criteria
    tasks = [
        {"name": "velocityRamp", "max_time": 74.08},
        {
            "name": "largeTimeStep",
            "max_time": 207.36,
            "std_limit": 1.0,
            "slope_limit": 0.25,
            "interval": 1.00,
            "use_oscillation_detection": True,
            "oscillation_quantities": ["Fx"],
            "oscillation_cutoff_freq": 0.2,
            "use_slope_criteria": False
        },
        {
            "name": "propulsionVariation",
            "max_time": 259.2,
            "std_limit": 0.5,
            "slope_limit": 0.125,
            "interval": 1.0,
            "use_oscillation_detection": True,
            "oscillation_quantities": ["Fx", "WFT"],
            "oscillation_cutoff_freq": 0.15,
            "use_slope_criteria": False
        },
        {
            "name": "final",
            "max_time": 259.2,
            "std_limit": 0.5,
            "slope_limit": 0.125,
            "interval": 1.0,
            "use_oscillation_detection": True,
            "oscillation_quantities": ["Fx", "WFT"],
            "oscillation_cutoff_freq": 0.15,
            "use_slope_criteria": False
        },
    ]

    # Define required convergence for each task
    required_convergence = {
        "largeTimeStep": ["Fx"],  # largeTimeStep only checks Fx
        "propulsionVariation": ["Fx", "WFT"],  # propulsionVariation requires BOTH Fx AND WFT to converge
        "final": ["Fx", "WFT"]  # final also requires BOTH Fx AND WFT to converge
    }

    # Create output directory for plots
    output_dir = "./convergence_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Run for a single simulation if testing
    if False:  # Set to True for testing a single simulation
        analyzer = ConvergenceAnalyzer()

        # Specify the simulation folder to test
        sim_folder = "./path/to/your/simulation"

        data_dict = {
            "Fx": analyzer.load_data(f"{sim_folder}/sixDoF/HULL/0/forces.dat", time_col=0, data_col=1),
            "WFT": analyzer.load_data(f"{sim_folder}/postProcessing/wake/0/volFieldValue.dat", time_col=0, data_col=1),
        }

        all_transitions, actual_task_ranges, time_saving, value_difference = analyzer.analyze_convergence_multi_quantity(
            data_dict,
            tasks,
            required_convergence,
            phase_summary_file=f"{sim_folder}/watch.log",
            original_results=f"{sim_folder}/CWBot.results",
            plot_results=True,
            save_plots=True,
            output_dir=f"{sim_folder}/plots"
        )
    else:
        # Initialize multi-simulation analyzer
        multi_analyzer = MultiSimulationAnalyzer()

        # Find simulation folders
        multi_analyzer.find_simulation_folders()

        # Process all simulations with plotting enabled
        results = multi_analyzer.process_all_simulations(
            tasks,
            required_convergence,
            plot_results=True,
            save_plots=True,
            output_dir=output_dir
        )

        # Export results to CSV
        multi_analyzer.export_results_to_csv()