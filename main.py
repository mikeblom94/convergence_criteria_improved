import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from collections import defaultdict
from scipy import signal


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

            import matplotlib.pyplot as plt

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(times, filtered_data, marker='o', linestyle='-')
            # Labels and title
            plt.xlabel('Time')
            plt.ylabel('Oscillation Value')
            plt.title('Oscillation filtered data Over Time')
            plt.grid(True)
            # Show plot
            plt.show()

        except Exception as e:
            print(f"Warning: Filtering failed: {e}")
            return None, None, values

        # Find zero crossings to identify oscillations (centered around mean)
        zero_crossings = np.where(np.diff(np.signbit(filtered_data - np.mean(filtered_data))))[0]
        print(f"Zero crossings: {zero_crossings}")
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

        if start_idx:
            import matplotlib.pyplot as plt

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(times[start_idx:end_idx], values[start_idx:end_idx], marker='o', linestyle='-')
            # Labels and title
            plt.xlabel('Time')
            plt.ylabel('Oscillation Value')
            plt.title('Oscillation Values Over Time')
            plt.grid(True)
            # Show plot
            plt.show()
            print("hi")

        # Initialize statistics dictionary
        stats = {}

        if start_idx is None or end_idx is None:
            # No clear oscillation detected, use fallback (return None to indicate fallback needed)
            return None

        # Calculate statistics over the detected oscillation
        oscillation_times = times[start_idx:end_idx + 1]
        oscillation_values = values[start_idx:end_idx + 1]

        # Mean using trapezoidal integration for better accuracy
        # Note: use trapezoid instead of trapz to avoid deprecation warning
        try:
            from scipy import integrate
            mean = integrate.trapezoid(oscillation_values, x=oscillation_times) / (oscillation_times[-1] - oscillation_times[0])
        except ImportError:
            # Fallback to numpy if scipy.integrate is not available
            mean = np.trapz(oscillation_values, x=oscillation_times) / (oscillation_times[-1] - oscillation_times[0])

        # Standard deviation over the oscillation
        std = np.std(oscillation_values, ddof=1)

        import matplotlib.pyplot as plt

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(oscillation_times, oscillation_values, marker='o', linestyle='-')
        # Labels and title
        plt.xlabel('Time')
        plt.ylabel('Oscillation Value')
        plt.title('Oscillation Values Over Time')
        plt.grid(True)
        # Show plot
        plt.show()

        # Calculate slope over the entire dataset for trend analysis
        # We use the entire dataset to see the overall trend, not just the oscillation
        lineregress = np.polyfit(times, values, 1)
        slope = lineregress[0]
        diff_height = slope * (times[-1] - times[0])

        # Store filtered data along with the times that correspond to it
        # This way we can properly plot it later
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
                          detect_oscillations=False, cutoff_freq=0.2, attempt_number=0):
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
                    # Don't print here, let the calling function handle logging
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
        diff_converged = abs(diff_height) <= limit_diff_he

        # Both criteria must be met for convergence
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

        return converged, stats, oscillation_detected

    def analyze_convergence_multi_quantity(self, data_dict, tasks, required_convergence=None):
        """
        Analyze convergence across multiple quantities with properly implemented progressive interval
        """
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
            end_time = current_time + max_time if max_time else current_time

            # Store the initial task range
            actual_task_ranges[task_name] = {"start": current_time, "end": end_time}

            # Special handling for reduceTimeStep (just a transition, not a convergence check)
            if task_name == "reduceTimeStep":
                current_time = end_time
                continue

            # If we need to check convergence
            if all(k in task for k in ['std_limit', 'slope_limit', 'interval']):
                # Get the required quantities for this task
                required_quantities = required_convergence.get(task_name, list(data_dict.keys()))
                print(f"  Required quantities for convergence: {required_quantities}")

                # Flag to track if any quantity converged
                task_converged = False
                convergence_time = None

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
                    print(f"    - Slope limit: {task['slope_limit']}%")
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
                            attempt_number=attempt
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
                                    print(f"    Stats: mean={stats['mean']:.2f}, std={stats['std_relative']:.4f}%, slope={stats['diff_height_relative']:.4f}%")

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
                                print(f"    Stats: mean={stats['mean']:.2f}, std={stats['std_relative']:.4f}%, slope={stats['diff_height_relative']:.4f}%")

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
                    print(f"  Task {task_name} converged early at {convergence_time}, skipping ahead to next task")

                    # Update the actual end time for this task
                    actual_task_ranges[task_name]["end"] = convergence_time

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

        # This return statement is now correctly outside the for loop
        return all_transitions, actual_task_ranges

    def plot_multi_quantity_results(self, data_dict, all_transitions, actual_task_ranges, output_prefix="convergence", shared_x_range=True):
        """
        Create visualizations for multiple quantities

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        output_prefix : str, optional
            Prefix for output files
        shared_x_range : bool, optional
            Whether to use the same x-range for all plots
        """
        # Find global min/max times if sharing x-range
        if shared_x_range:
            global_min_time = float('inf')
            global_max_time = float('-inf')

            for times, _ in data_dict.values():
                if len(times) > 0:
                    global_min_time = min(global_min_time, times[0])
                    global_max_time = max(global_max_time, times[-1])

        # Create output directory if it doesn't exist
        os.makedirs("./plots", exist_ok=True)

        # Plot each quantity
        for quantity, (times, values) in data_dict.items():
            transitions = all_transitions[quantity]

            self.plot_single_quantity(
                quantity, times, values, transitions, actual_task_ranges,
                f"{output_prefix}_{quantity.lower()}.png",
                x_range=(global_min_time, global_max_time) if shared_x_range else None
            )

    def plot_oscillation_comparison(self, data_dict, all_transitions, task_name, output_prefix="oscillation_comparison"):
        """
        Create visualizations specifically showing oscillation detection for quantities
        that had oscillation detection enabled.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        task_name : str
            Name of task to focus on
        output_prefix : str, optional
            Prefix for output files
        """
        os.makedirs("./plots", exist_ok=True)

        # Find all quantities that used oscillation detection
        for quantity, transitions in all_transitions.items():
            # Find convergence events that used oscillation detection
            for trans in transitions:
                if (trans["type"] == "converge" and
                        trans["task"] == task_name and
                        trans.get("stats", {}).get("used_oscillation_detection", False)):
                    times, values = data_dict[quantity]
                    stats = trans["stats"]

                    # Create the oscillation comparison plot
                    self._plot_oscillation_detection(
                        quantity, times, values, stats,
                        f"{output_prefix}_{task_name}_{quantity.lower()}.png"
                    )

    def _plot_oscillation_detection(self, quantity, times, values, stats, output_file):
        """Helper method to plot oscillation detection details"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

        # Plot the full data
        ax1.plot(times, values, 'b-', alpha=0.5, label='Raw data')

        # If filtered data is available, plot it
        # Only use the portion of times that corresponds to the detected oscillation
        if 'filtered_data' in stats:
            # Make sure we're only plotting the filtered data for the range we analyzed
            # This fixes the dimension mismatch error
            try:
                if 'begin' in stats and 'end' in stats:
                    begin = stats['begin']
                    end = stats['end']

                    # Find indices corresponding to oscillation window
                    window_mask = (times >= begin) & (times <= end)
                    window_times = times[window_mask]
                    window_values = values[window_mask]

                    # Apply filtering to this window only
                    if len(window_times) > 4:  # Minimum length for filtering
                        # Calculate sampling frequency based on this window
                        dt = np.mean(np.diff(window_times))
                        sampling_freq = 1.0 / dt

                        # Get cutoff frequency (use same as in original detection)
                        cutoff_freq = 0.2  # Default value
                        if 'cutoff_freq' in stats:
                            cutoff_freq = stats['cutoff_freq']

                        # Apply filter to window
                        nyquist = 0.5 * sampling_freq
                        normalized_cutoff = cutoff_freq / nyquist
                        b, a = signal.butter(4, normalized_cutoff, btype='low')
                        window_filtered = signal.filtfilt(b, a, window_values)

                        # Plot filtered data for window
                        ax1.plot(window_times, window_filtered, 'g-', label='Filtered data')

                # If we have stored filtered data with times, use that
                elif 'filtered_times' in stats and len(stats['filtered_times']) == len(stats['filtered_data']):
                    ax1.plot(stats['filtered_times'], stats['filtered_data'], 'g-', label='Filtered data')
            except Exception as e:
                print(f"Warning: Could not plot filtered data: {e}")

        # Highlight the oscillation window
        begin = stats['begin']
        end = stats['end']
        ax1.axvspan(begin, end, alpha=0.3, color='yellow', label='Oscillation window')

        # Plot the mean line
        mean = stats['mean']
        ax1.axhline(y=mean, color='r', linestyle='-', label=f'Mean: {mean:.3f}')

        # Add convergence info
        ax1.set_title(f"{quantity} - Oscillation Detection Analysis")
        ax1.legend(loc='best')
        ax1.grid(True)

        # Detail plot of the oscillation
        if 'cut_times' in stats and 'cut_values' in stats:
            cut_times = stats['cut_times']
            cut_values = stats['cut_values']

            ax2.plot(cut_times, cut_values, 'b-', label='Data in oscillation window')
            ax2.axhline(y=mean, color='r', linestyle='-', label=f'Mean: {mean:.3f}')

            # Add standard deviation band
            std = stats['std']
            std_rel = stats.get('std_relative', abs((std / mean) * 100))
            diff_height_rel = stats.get('diff_height_relative', abs((stats['diff_height'] / mean) * 100))

            ax2.fill_between(cut_times, mean - std, mean + std, alpha=0.2, color='gray',
                             label=f'Std: {std:.3f} ({std_rel:.2f}%)')

            ax2.set_title(f"Detail of detected oscillation (Slope: {diff_height_rel:.2f}%)")
            ax2.legend(loc='best')
            ax2.grid(True)

            # Add textbox with stats
            textstr = f"Mean: {mean:.3f}\nStd: {std:.3f} ({std_rel:.2f}%)\nSlope: {diff_height_rel:.2f}%"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

        plt.tight_layout()

        if output_file:
            plt.savefig(f"./plots/{output_file}", dpi=300, bbox_inches='tight')
            print(f"Oscillation detection plot saved to {output_file}")

        plt.close(fig)

    def plot_single_quantity(self, quantity, times, values, transitions, actual_task_ranges, output_file=None, x_range=None):
        """
        Create visualization for a single quantity

        Parameters:
        -----------
        quantity : str
            Name of the quantity being analyzed
        times : array
            Time points
        values : array
            Data values
        transitions : list
            Transitions for this quantity
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        output_file : str, optional
            Path to save the plot
        x_range : tuple, optional
            (min, max) values for x-axis
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1], sharex=True)

        # Plot the full data
        ax1.plot(times, values, 'b-', label='Data')

        # Set title with quantity name
        ax1.set_title(f"{quantity} - OpenFOAM Convergence Analysis")

        # Add colored regions for tasks based on actual ranges
        task_colors = plt.cm.tab10.colors
        color_idx = 0

        # Plot the actual task ranges
        for task_name, time_range in actual_task_ranges.items():
            start_time = time_range["start"]
            end_time = time_range["end"]
            task_color = task_colors[color_idx % len(task_colors)]

            ax1.axvspan(start_time, end_time, alpha=0.2, color=task_color,
                        label=f"Task: {task_name}")
            color_idx += 1

        # Add convergence points
        for t in transitions:
            if t["type"] == "converge":
                ax1.axvline(x=t["time"], color='r', linestyle='--')

                # Get detection method
                detection_method = t.get("detection_method", "")
                annotation_text = f"Converged: {t['task']}"
                if detection_method:
                    annotation_text += f" ({detection_method})"

                ax1.annotate(annotation_text,
                             xy=(t["time"], np.max(values)),
                             xytext=(t["time"], np.max(values) + 0.05 * (np.max(values) - np.min(values))),
                             arrowprops=dict(arrowstyle="->", color='r'))

                # If we have stats, add statistics box
                if "stats" in t:
                    stats = t["stats"]
                    mean = stats['mean']
                    std = stats.get('std_relative', 0)
                    diff = stats.get('diff_height_relative', 0)

                    # Add convergence window
                    converge_mask = (times >= stats['begin']) & (times <= stats['end'])
                    ax1.axvspan(stats['begin'], stats['end'], alpha=0.3, color='g')

                    # Add stats values
                    textstr = f"Mean: {mean:.3f}\nStd: {std:.2f}%\nSlope: {diff:.2f}%"
                    if stats.get('used_oscillation_detection', False):
                        textstr += "\n(Using oscillation detection)"
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax1.text(stats['end'], mean, textstr, fontsize=9,
                             verticalalignment='center', bbox=props)

                    # Add mean line
                    ax1.plot([stats['begin'], stats['end']], [mean, mean], 'r-', linewidth=1.5)

        # Add task transitions to second subplot as a Gantt-like chart
        current_tasks = []
        y_positions = {}
        y_pos = 0

        task_spans = {}
        for t in transitions:
            task = t["task"]

            if t["type"] == "start":
                if task not in task_spans:
                    task_spans[task] = {"start": t["time"]}

                if task not in y_positions:
                    y_positions[task] = y_pos
                    y_pos += 1

            elif t["type"] == "end":
                if task in task_spans:
                    task_spans[task]["end"] = t["time"]
                    task_spans[task]["reason"] = t.get("reason", "unknown")

        # Draw Gantt chart based on consolidated spans
        for task, span in task_spans.items():
            if "start" in span and "end" in span:
                start_time = span["start"]
                end_time = span["end"]
                reason = span.get("reason", "unknown")

                # Get the task color
                task_color = task_colors[list(task_spans.keys()).index(task) % len(task_colors)]

                # Draw task bar
                ax2.barh([y_positions[task]], width=end_time - start_time, left=start_time,
                         color=task_color, alpha=0.7)

                # Add task name
                ax2.text(start_time + (end_time - start_time) / 2, y_positions[task],
                         task, ha='center', va='center', color='black', fontweight='bold')

                # Add star if converged
                if reason == "converged":
                    ax2.plot(end_time, y_positions[task], 'r*', markersize=10)

        # Set x-range if specified
        if x_range is not None:
            ax1.set_xlim(x_range)

        # Set labels
        ax1.set_ylabel(quantity)
        ax1.grid(True)

        ax2.set_yticks(list(y_positions.values()))
        ax2.set_yticklabels(list(y_positions.keys()))
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Tasks')
        ax2.grid(True, axis='x')

        # Add legend to top plot
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()

        if output_file:
            plt.savefig(f"./plots/{output_file}", dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")

        plt.close(fig)

    def print_multi_quantity_summary(self, all_transitions, actual_task_ranges):
        """Print a summary of the simulation progression for all quantities"""
        # Get all unique tasks across all quantities
        all_tasks = set()
        for transitions in all_transitions.values():
            for t in transitions:
                if t["type"] == "start":
                    all_tasks.add(t["task"])

        # Sort tasks by their start times
        sorted_tasks = sorted(all_tasks, key=lambda x: actual_task_ranges[x]["start"])

        # Print summary for each task
        print("\nSimulation progression summary:")
        for task in sorted_tasks:
            # Get actual task range
            start_time = actual_task_ranges[task]["start"]
            end_time = actual_task_ranges[task]["end"]
            task_duration = end_time - start_time

            print(f"Time {start_time:.2f}s: Started {task}")

            # Find convergence information for each quantity
            for quantity, transitions in all_transitions.items():
                # Find convergence for this task and quantity
                convergence = next((t for t in transitions if
                                    t["type"] == "converge" and t["task"] == task), None)

                if convergence:
                    stats = convergence.get("stats", {})
                    std = stats.get('std_relative', 'N/A')
                    diff = stats.get('diff_height_relative', 'N/A')
                    detection_method = "oscillation detection" if stats.get('used_oscillation_detection', False) else "fixed interval"
                    print(f"  Time {convergence['time']:.2f}s: {quantity} converged using {detection_method} (Std: {std:.2f}%, Slope: {diff:.2f}%)")

            print(f"Time {end_time:.2f}s: {task} ended (Duration: {task_duration:.2f}s)")
            print("")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ConvergenceAnalyzer()

    # Example configuration with oscillation detection
    tasks = [
        {"name": "velocityRamp", "max_time": 74.08},
        {
            "name": "largeTimeStep",
            "max_time": 207.36,
            "std_limit": 1.0,
            "slope_limit": 0.25,
            "interval": 10.00,
            "use_oscillation_detection": True,  # Enable oscillation detection
            "oscillation_quantities": ["Fx"],  # Apply to Fx only
            "oscillation_cutoff_freq": 0.2  # Cutoff frequency in Hz
        },
        {"name": "reduceTimeStep"},
        {
            "name": "propulsionVariation",
            "max_time": 259.2,
            "std_limit": 0.5,
            "slope_limit": 0.125,
            "interval": 10.0,
            "use_oscillation_detection": True,  # Enable oscillation detection
            "oscillation_quantities": ["Fx", "WFT"],  # Apply to both Fx and WFT
            "oscillation_cutoff_freq": 0.15  # Different cutoff frequency
        },
        {
            "name": "final",
            "max_time": 259.2,
            "std_limit": 0.5,
            "slope_limit": 0.125,
            "interval": 10.0,
            "use_oscillation_detection": True,  # Enable oscillation detection
            "oscillation_quantities": ["Fx", "WFT"],  # Apply to both Fx and WFT
            "oscillation_cutoff_freq": 0.01  # Different cutoff frequency
        },
        {"name": "stopJob"},
    ]

    # Define required convergence for each task
    required_convergence = {
        "largeTimeStep": ["Fx"],  # largeTimeStep only checks Fx
        "propulsionVariation": ["Fx", "WFT"],  # propulsionVariation requires BOTH Fx AND WFT to converge
        "final": ["Fx", "WFT"]  # final also requires BOTH Fx AND WFT to converge
    }

    try:
        # Load data for each quantity
        data_dict = {
            "Fx": analyzer.load_data("sixDoF/HULL/0/forces.dat", time_col=0, data_col=1),
            "FxP": analyzer.load_data("sixDoF/HULL/0/forcesPV.dat", time_col=0, data_col=1),
            "FxV": analyzer.load_data("sixDoF/HULL/0/forcesPV.dat", time_col=0, data_col=7),
            "Trim": analyzer.load_data("sixDoF/HULL/0/position.dat", time_col=0, data_col=5),
            "Sinkage": analyzer.load_data("postProcessing/sinkage/0/relativeMotion.dat", time_col=0, data_col=3),
            "WFT": analyzer.load_data("postProcessing/wake/0/volFieldValue.dat", time_col=0, data_col=1),
            "Wetted": analyzer.load_data("postProcessing/wettedSurface/0/surfaceFieldValue.dat", time_col=0, data_col=1)
        }

        # Analyze convergence across all quantities
        all_transitions, actual_task_ranges = analyzer.analyze_convergence_multi_quantity(
            data_dict,
            tasks,
            required_convergence
        )

        # Print summary of convergence
        analyzer.print_multi_quantity_summary(all_transitions, actual_task_ranges)

        # Plot results for each quantity
        analyzer.plot_multi_quantity_results(data_dict, all_transitions, actual_task_ranges, "convergence_with_oscillation")

        # Plot detailed oscillation detection visualizations
        analyzer.plot_oscillation_comparison(data_dict, all_transitions, "largeTimeStep")
        analyzer.plot_oscillation_comparison(data_dict, all_transitions, "propulsionVariation")
        analyzer.plot_oscillation_comparison(data_dict, all_transitions, "final")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Falling back to synthetic data for demonstration...")
