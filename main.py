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

            # import matplotlib.pyplot as plt
            #
            # # Plot
            # plt.figure(figsize=(10, 5))
            # plt.plot(times, filtered_data, marker='o', linestyle='-')
            # # Labels and title
            # plt.xlabel('Time')
            # plt.ylabel('Oscillation Value')
            # plt.title('Oscillation filtered data Over Time')
            # plt.grid(True)
            # # Show plot
            # plt.show()

        except Exception as e:
            print(f"Warning: Filtering failed: {e}")
            return None, None, values

        # Find zero crossings to identify oscillations (centered around mean)
        zero_crossings = np.where(np.diff(np.signbit(filtered_data - np.mean(filtered_data))))[0]
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

        # if start_idx:
        #     import matplotlib.pyplot as plt
        #
        #     # Plot
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(times[start_idx:end_idx], values[start_idx:end_idx], marker='o', linestyle='-')
        #     # Labels and title
        #     plt.xlabel('Time')
        #     plt.ylabel('Oscillation Value')
        #     plt.title('Oscillation Values Over Time')
        #     plt.grid(True)
        #     # Show plot
        #     plt.show()

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

        # import matplotlib.pyplot as plt
        #
        # # Plot
        # plt.figure(figsize=(10, 5))
        # plt.plot(oscillation_times, oscillation_values, marker='o', linestyle='-')
        # # Labels and title
        # plt.xlabel('Time')
        # plt.ylabel('Oscillation Value')
        # plt.title('Oscillation Values Over Time')
        # plt.grid(True)
        # # Show plot
        # plt.show()

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
                          detect_oscillations=False, cutoff_freq=0.2, attempt_number=0,
                          use_slope_criteria=True):  # Added parameter to toggle slope criteria
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
        stats['use_slope_criteria'] = use_slope_criteria  # Add flag to stats for reference

        return converged, stats, oscillation_detected

    # Add the new function to load and process the CSV file
    def load_original_phase_summary(self, csv_file="original_phase_summary.csv"):
        """
        Load and process the original phase summary CSV file.

        Parameters:
        -----------
        csv_file : str
            Path to the CSV file

        Returns:
        --------
        dict : Dictionary organized by quantity and phase, containing phase information
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Original phase summary file not found: {csv_file}")

        # Load the CSV file using pandas
        df = pd.read_csv(csv_file)

        # Initialize a nested dictionary to store the data
        phase_summary = {}

        # Process each row in the CSV
        for _, row in df.iterrows():
            quantity = row['Quantity']
            phase = row['Phase']

            # Initialize the quantity dictionary if it doesn't exist
            if quantity not in phase_summary:
                phase_summary[quantity] = {}

            # Check if values are strings and handle special cases
            converged = row['Converged'] == 'Yes' if isinstance(row['Converged'], str) else row['Converged']
            conv_time = row['Convergence Time']
            conv_value = row['Convergence Value']

            # Handle N/A or NA string values
            if isinstance(conv_time, str) and conv_time.upper() in ['N/A', 'NA']:
                conv_time = None
            if isinstance(conv_value, str) and conv_value.upper() in ['N/A', 'NA']:
                conv_value = None

            # Store the phase data
            phase_summary[quantity][phase] = {
                'start_time': row['Start Time'],
                'end_time': row['End Time'],
                'start_value': row['Start Value'],
                'end_value': row['End Value'],
                'converged': converged,
                'convergence_time': conv_time if converged else None,
                'convergence_value': conv_value if converged else None,
                'time_saved': row['Time Saved']
            }

        return phase_summary

    def analyze_convergence_multi_quantity(self, data_dict, tasks, required_convergence=None):
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

        Returns:
        --------
        tuple : (all_transitions, actual_task_ranges)
            Transitions for all quantities and actual task time ranges
        """

        # Load original phase summary from CSV
        original_phase_summary = analyzer.load_original_phase_summary("original_phase_summary.csv")

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

            if task_name == "final":
                print(f"\n=== Analyzing task: {task_name} ===")

            print(f"\n=== Analyzing task: {task_name} ===")
            print(f"  Start time: {current_time}")

            if task_name == "propulsionVariation":
                current_time = max(original_phase_summary['Fx']["largeTimeStep"]["end_time"], original_phase_summary['WFT']["largeTimeStep"]["end_time"])
            if task_name == "final":
                current_time = max(original_phase_summary['Fx']["propulsionVariation"]["end_time"], original_phase_summary['WFT']["propulsionVariation"]["end_time"])

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

        # Time saving calculation
        # Get all phases in order
        phases = list(actual_task_ranges.keys())

        # Calculate time saving between consecutive phases
        time_saving = 0
        for i in range(len(phases) - 1):
            current_phase = phases[i]
            next_phase = phases[i + 1]

            current_end = actual_task_ranges[current_phase]['end']
            next_start = actual_task_ranges[next_phase]['start']

            # If next phase starts after current phase ends, that's time saved
            if next_start > current_end:
                time_saving += next_start - current_end

        # Return statement is outside the for loop
        return all_transitions, actual_task_ranges, time_saving


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ConvergenceAnalyzer()

    # Example configuration with oscillation detection and optional slope criteria
    tasks = [
        {"name": "velocityRamp", "max_time": 74.08},
        {
            "name": "largeTimeStep",
            "max_time": 207.36,
            "std_limit": 1.0,
            "slope_limit": 0.25,  # Still needed even if use_slope_criteria is False
            "interval": 10.00,
            "use_oscillation_detection": True,
            "oscillation_quantities": ["Fx"],
            "oscillation_cutoff_freq": 0.2,
            "use_slope_criteria": False  # Disabling slope criteria for this task
        },
        {"name": "reduceTimeStep"},
        {
            "name": "propulsionVariation",
            "max_time": 259.2,
            "std_limit": 0.5,
            "slope_limit": 0.125,
            "interval": 10.0,
            "use_oscillation_detection": True,
            "oscillation_quantities": ["Fx", "WFT"],
            "oscillation_cutoff_freq": 0.15,
            "use_slope_criteria": False  # Disabling slope criteria for this task
        },
        {
            "name": "final",
            "max_time": 259.2,
            "std_limit": 0.5,
            "slope_limit": 0.125,
            "interval": 10.0,
            "use_oscillation_detection": True,
            "oscillation_quantities": ["Fx", "WFT"],
            "oscillation_cutoff_freq": 0.15,
            "use_slope_criteria": False  # Disabling slope criteria for this task
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
        all_transitions, actual_task_ranges, time_saving = analyzer.analyze_convergence_multi_quantity(
            data_dict,
            tasks,
            required_convergence
        )

        print(f"Time saving: {time_saving:.2f} seconds")
        print(f"That is {time_saving / actual_task_ranges['final']['end']:.2%} of the total simulation time")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Falling back to synthetic data for demonstration...")