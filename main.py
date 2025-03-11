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
        # Return statement is outside the for loop
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
                    textstr = f"Mean: {mean:.3f}\nStd: {std:.2f}%"

                    # Check if slope criteria was used
                    if stats.get('use_slope_criteria', True):
                        textstr += f"\nSlope: {diff:.2f}%"
                    else:
                        textstr += "\nSlope criteria disabled"

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

                    # Check if slope criteria was used
                    slope_criteria_used = stats.get('use_slope_criteria', True)

                    print(f"  Time {convergence['time']:.2f}s: {quantity} converged using {detection_method}")
                    print(f"    - Std: {std:.2f}%")

                    # Only print slope info if it was used in convergence check
                    if slope_criteria_used:
                        print(f"    - Slope: {diff:.2f}%")
                    else:
                        print(f"    - Slope criteria disabled")

            print(f"Time {end_time:.2f}s: {task} ended (Duration: {task_duration:.2f}s)")
            print("")

    def create_optimized_data(self, data_dict, all_transitions, actual_task_ranges, tasks):
        """
        Create optimized data sets that remove periods between convergence and next phase start.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        tasks : list
            Original task configurations

        Returns:
        --------
        dict: Dictionary mapping quantity names to optimized (times, values) tuples
        """
        # Dictionary to store optimized data for each quantity
        optimized_data = {}

        # Get ordered list of tasks
        ordered_tasks = []
        for task in tasks:
            if task["name"] not in ["stopJob"]:  # Exclude end tasks with no convergence criteria
                ordered_tasks.append(task["name"])

        # Process each quantity
        for quantity, (times, values) in data_dict.items():
            print(f"\nProcessing optimized data for {quantity}...")

            # Initialize lists for optimized data
            opt_times = []
            opt_values = []

            # Get transitions for this quantity
            transitions = all_transitions[quantity]

            # Process each task in order
            for i, task_name in enumerate(ordered_tasks):
                print(f"  Analyzing task: {task_name}")

                # Get task start time
                task_start = actual_task_ranges[task_name]["start"]

                # Find convergence for this task
                convergence = next((t for t in transitions if
                                    t["type"] == "converge" and t["task"] == task_name), None)

                if convergence:
                    convergence_time = convergence["time"]
                    print(f"    Task converged at {convergence_time:.2f}s")

                    # Find data points from task start to convergence
                    mask = (times >= task_start) & (times <= convergence_time)
                    task_times = times[mask]
                    task_values = values[mask]

                    # Add this section to optimized data
                    opt_times.extend(task_times)
                    opt_values.extend(task_values)

                    # If this isn't the last task, add a marker point
                    if i < len(ordered_tasks) - 1:
                        next_task = ordered_tasks[i + 1]
                        next_start = actual_task_ranges[next_task]["start"]

                        # Add a marker point to show the jump
                        if next_start > convergence_time:
                            # Create a small gap in the data to clearly show the transition
                            opt_times.append(convergence_time + 0.001)  # Small increment
                            opt_values.append(None)  # None creates a break in the plot line

                            # Add the next task's start point
                            opt_times.append(next_start)

                            # Interpolate the value at next_start
                            next_start_mask = (times >= next_start)
                            if np.any(next_start_mask):
                                next_start_idx = np.argmax(next_start_mask)
                                next_value = values[next_start_idx]
                                opt_values.append(next_value)
                            else:
                                # Use the last value if we can't find the next
                                opt_values.append(task_values[-1])

                            print(f"    Jumped from {convergence_time:.2f}s to {next_start:.2f}s (skipped {next_start - convergence_time:.2f}s)")
                else:
                    # If no convergence found, include data up to the end of the task
                    task_end = actual_task_ranges[task_name]["end"]
                    print(f"    Task did not converge, including all data up to {task_end:.2f}s")

                    # Find data points from task start to task end
                    mask = (times >= task_start) & (times <= task_end)
                    task_times = times[mask]
                    task_values = values[mask]

                    # Add this section to optimized data
                    opt_times.extend(task_times)
                    opt_values.extend(task_values)

                    # Add a marker for the next task if applicable
                    if i < len(ordered_tasks) - 1:
                        next_task = ordered_tasks[i + 1]
                        next_start = actual_task_ranges[next_task]["start"]

                        # Create a small gap in the data to clearly show the transition
                        opt_times.append(task_end + 0.001)  # Small increment
                        opt_values.append(None)  # None creates a break in the plot line

                        # Add the next task's start point
                        opt_times.append(next_start)

                        # Interpolate the value at next_start
                        next_start_mask = (times >= next_start)
                        if np.any(next_start_mask):
                            next_start_idx = np.argmax(next_start_mask)
                            next_value = values[next_start_idx]
                            opt_values.append(next_value)
                        else:
                            # Use the last value if we can't find the next
                            opt_values.append(task_values[-1])

            # Convert to numpy arrays
            optimized_data[quantity] = (np.array(opt_times), np.array(opt_values))

            # Calculate time savings
            original_duration = times[-1] - times[0]
            optimized_duration = sum([opt_times[i + 1] - opt_times[i] for i in range(len(opt_times) - 1)
                                      if opt_values[i] is not None and opt_values[i + 1] is not None])
            time_saved = original_duration - optimized_duration
            print(f"  Time saved for {quantity}: {time_saved:.2f}s ({(time_saved / original_duration) * 100:.1f}% reduction)")

        return optimized_data

    def plot_optimized_comparison(self, data_dict, optimized_data, all_transitions, actual_task_ranges, output_prefix="optimized"):
        """
        Create comparison plots showing original vs optimized data with improved visualization

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to original (times, values) tuples
        optimized_data : dict
            Dictionary mapping quantity names to optimized (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        output_prefix : str, optional
            Prefix for output files
        """
        # Create output directory if it doesn't exist
        os.makedirs("./plots", exist_ok=True)

        # Plot each quantity
        for quantity in data_dict.keys():
            original_times, original_values = data_dict[quantity]
            opt_times, opt_values = optimized_data[quantity]
            transitions = all_transitions[quantity]

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])

            # Plot original data
            ax1.plot(original_times, original_values, 'b-', label='Original Data')
            ax1.set_title(f"{quantity} - Original Data")

            # Find time jumps in optimized data (where None values are)
            jump_indices = [i for i, v in enumerate(opt_values) if v is None]
            continuous_segments = []
            start_idx = 0

            # Create continuous segments for plotting
            for jump_idx in jump_indices:
                if jump_idx > start_idx:
                    # Add segment from start_idx to jump_idx-1
                    continuous_segments.append((start_idx, jump_idx))
                # Next segment starts after the None value
                start_idx = jump_idx + 1

            # Add the final segment if there's data after the last jump
            if start_idx < len(opt_times):
                continuous_segments.append((start_idx, len(opt_times)))

            # Plot optimized data by segments with different colors to clearly show jumps
            segment_colors = ['g', 'darkgreen', 'limegreen', 'forestgreen', 'seagreen']
            for i, (start_idx, end_idx) in enumerate(continuous_segments):
                color = segment_colors[i % len(segment_colors)]
                segment_times = opt_times[start_idx:end_idx]
                segment_values = opt_values[start_idx:end_idx]

                # Skip if segment is empty
                if len(segment_times) == 0:
                    continue

                ax2.plot(segment_times, segment_values, color=color, linestyle='-',
                         label=f"Segment {i + 1}" if i < 5 else "")

            # Convert opt_values to a numpy array, filtering out None values
            # This is for calculating min/max safely
            non_none_values = [v for v in opt_values if v is not None]

            # Make sure we have at least one non-None value
            if non_none_values:
                # Calculate min/max for y-axis positioning
                min_val = min(non_none_values)
                max_val = max(non_none_values)
                mid_y = (min_val + max_val) / 2

                # Also plot jump markers (vertical dashed lines to show skips)
                for jump_idx in jump_indices:
                    if jump_idx > 0 and jump_idx < len(opt_times) - 1:
                        jump_time_before = opt_times[jump_idx - 1]
                        jump_time_after = opt_times[jump_idx + 1]

                        # Add arrow showing the jump
                        ax2.annotate("",
                                     xy=(jump_time_after, mid_y),
                                     xytext=(jump_time_before, mid_y),
                                     arrowprops=dict(arrowstyle="->", color='red', lw=2),
                                     annotation_clip=False)

                        # Add text showing time saved
                        time_saved = jump_time_after - jump_time_before
                        ax2.text((jump_time_before + jump_time_after) / 2, mid_y,
                                 f"{time_saved:.2f}s\nskipped",
                                 color='red', fontweight='bold', ha='center', va='bottom')

            ax2.set_title(f"{quantity} - Optimized Data (With Time Jumps)")

            # Add colored regions for tasks based on actual ranges
            task_colors = plt.cm.tab10.colors
            color_idx = 0

            # Add task regions to both plots
            for task_name, time_range in actual_task_ranges.items():
                start_time = time_range["start"]
                end_time = time_range["end"]
                task_color = task_colors[color_idx % len(task_colors)]

                # Add to original plot
                ax1.axvspan(start_time, end_time, alpha=0.2, color=task_color,
                            label=f"Task: {task_name}")

                # Add to optimized plot (if visible in the window)
                # We need to check if any part of this task's range is in the optimized data
                if any((opt_times >= start_time) & (opt_times <= end_time)):
                    ax2.axvspan(start_time, end_time, alpha=0.2, color=task_color)

                color_idx += 1

            # Add convergence points to both plots
            for t in transitions:
                if t["type"] == "converge":
                    # Add to original plot
                    ax1.axvline(x=t["time"], color='r', linestyle='--')

                    # Add to optimized plot (if visible in the optimized window)
                    if any(opt_times >= t["time"]):
                        ax2.axvline(x=t["time"], color='r', linestyle='--')

                    # Add annotation to original plot
                    ax1.annotate(f"Converged: {t['task']}",
                                 xy=(t["time"], np.max(original_values)),
                                 xytext=(t["time"], np.max(original_values) + 0.05 * (np.max(original_values) - np.min(original_values))),
                                 arrowprops=dict(arrowstyle="->", color='r'))

            # Add legends
            ax1.legend(loc='upper right')
            handles, labels = ax2.get_legend_handles_labels()
            if len(handles) > 0:
                ax2.legend(handles[:5], labels[:5], loc='upper right')  # Limit to first 5 segments

            # Add grid
            ax1.grid(True)
            ax2.grid(True)

            # Set ylim the same for both plots for easier comparison
            # First ensure we're dealing with non-None values
            if non_none_values:
                min_val = min(non_none_values)
                max_val = max(non_none_values)

                ymin = min(np.min(original_values), min_val)
                ymax = max(np.max(original_values), max_val)
                buffer = (ymax - ymin) * 0.1
                ax1.set_ylim(ymin - buffer, ymax + buffer)
                ax2.set_ylim(ymin - buffer, ymax + buffer)

            # Calculate time savings
            original_duration = original_times[-1] - original_times[0]
            optimized_duration = sum([opt_times[i + 1] - opt_times[i] for i in range(len(opt_times) - 1)
                                      if i + 1 < len(opt_values) and opt_values[i] is not None and opt_values[i + 1] is not None])
            time_saved = original_duration - optimized_duration

            # Add time savings info
            fig.suptitle(f"{quantity} - Convergence Optimization Comparison\nTime saved: {time_saved:.2f}s ({(time_saved / original_duration) * 100:.1f}% reduction)",
                         fontsize=14)

            # Set labels
            ax1.set_ylabel(quantity)
            ax2.set_ylabel(quantity)
            ax2.set_xlabel('Time [s]')

            plt.tight_layout()

            # Save figure
            plt.savefig(f"./plots/{output_prefix}_{quantity.lower()}.png", dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output_prefix}_{quantity.lower()}.png")

            plt.close(fig)

    def create_compressed_timeline_data(self, data_dict, optimized_data, output_prefix="compressed"):
        """
        Create a version of the optimized data with a compressed timeline.
        This removes the time gaps completely and creates a continuous timeline.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to original (times, values) tuples
        optimized_data : dict
            Dictionary mapping quantity names to optimized (times, values) tuples
        output_prefix : str, optional
            Prefix for output files

        Returns:
        --------
        dict: Dictionary mapping quantity names to compressed (times, values) tuples
        """
        compressed_data = {}

        for quantity, (opt_times, opt_values) in optimized_data.items():
            print(f"\nCreating compressed timeline for {quantity}...")

            # Initialize compressed times and values
            compressed_times = []
            compressed_values = []

            # Find time jumps (None values)
            jump_indices = [i for i, v in enumerate(opt_values) if v is None]

            # Process data in segments
            current_time = 0.0  # Start at time 0
            last_valid_time = 0.0

            for i in range(len(opt_times)):
                if i in jump_indices:
                    # Skip None values
                    continue

                if opt_values[i] is not None:
                    if i > 0 and i - 1 in jump_indices:
                        # This is the first point after a jump
                        # Calculate how much time to skip
                        time_diff = opt_times[i] - last_valid_time
                        current_time += 0.01  # Just add a small increment to show the jump
                    else:
                        # Regular point - add the time difference from the last point
                        if len(compressed_times) > 0:
                            time_diff = opt_times[i] - opt_times[i - 1]
                            current_time += time_diff
                        else:
                            # First point - keep the original time
                            current_time = opt_times[i]

                    # Add point to compressed data
                    compressed_times.append(current_time)
                    compressed_values.append(opt_values[i])

                    # Update last valid time
                    last_valid_time = opt_times[i]

            # Convert to numpy arrays
            compressed_data[quantity] = (np.array(compressed_times), np.array(compressed_values))

            # Calculate compression ratio
            original_times, _ = data_dict[quantity]
            original_duration = original_times[-1] - original_times[0]

            # Make sure we have compressed data to work with
            if len(compressed_times) > 0:
                compressed_duration = compressed_times[-1] - compressed_times[0]

                print(f"  Original duration: {original_duration:.2f}s")
                print(f"  Compressed duration: {compressed_duration:.2f}s")
                print(f"  Compression ratio: {compressed_duration / original_duration:.2f}")
            else:
                print("  No compressed data generated")

        return compressed_data

    def plot_compressed_data(self, data_dict, optimized_data, compressed_data, all_transitions, actual_task_ranges, output_prefix="compressed"):
        """
        Create plots showing original, optimized with gaps, and fully compressed data

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to original (times, values) tuples
        optimized_data : dict
            Dictionary mapping quantity names to optimized (times, values) tuples
        compressed_data : dict
            Dictionary mapping quantity names to compressed (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        output_prefix : str, optional
            Prefix for output files
        """
        # Create output directory if it doesn't exist
        os.makedirs("./plots", exist_ok=True)

        # Plot each quantity
        for quantity in data_dict.keys():
            original_times, original_values = data_dict[quantity]
            opt_times, opt_values = optimized_data[quantity]
            compressed_times, compressed_values = compressed_data[quantity]

            # Skip if compressed data is empty
            if len(compressed_times) == 0:
                print(f"Skipping {quantity} - no compressed data")
                continue

            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), height_ratios=[1, 1, 1])

            # Plot original data
            ax1.plot(original_times, original_values, 'b-', label='Original Data')
            ax1.set_title(f"{quantity} - Original Data (Full Timeline)")

            # Plot optimized data with gaps
            # Convert None values to NaN for plotting
            opt_values_plot = np.array([float('nan') if v is None else v for v in opt_values])
            ax2.plot(opt_times, opt_values_plot, 'g-', label='Optimized Data (With Gaps)')
            ax2.set_title(f"{quantity} - Optimized Data (With Time Gaps)")

            # Plot compressed data
            ax3.plot(compressed_times, compressed_values, 'r-', label='Compressed Data (Continuous)')
            ax3.set_title(f"{quantity} - Compressed Timeline (No Gaps)")

            # Add legends
            ax1.legend(loc='upper right')
            ax2.legend(loc='upper right')
            ax3.legend(loc='upper right')

            # Add grid
            ax1.grid(True)
            ax2.grid(True)
            ax3.grid(True)

            # Extract non-None values for min/max calculation
            non_none_values = [v for v in opt_values if v is not None]

            # Set ylim the same for all plots
            if non_none_values:
                ymin = min(np.min(original_values), min(non_none_values), np.min(compressed_values))
                ymax = max(np.max(original_values), max(non_none_values), np.max(compressed_values))
                buffer = (ymax - ymin) * 0.1
                ax1.set_ylim(ymin - buffer, ymax + buffer)
                ax2.set_ylim(ymin - buffer, ymax + buffer)
                ax3.set_ylim(ymin - buffer, ymax + buffer)

            # Calculate time savings
            original_duration = original_times[-1] - original_times[0]
            compressed_duration = compressed_times[-1] - compressed_times[0]
            time_saved = original_duration - compressed_duration

            # Add time savings info
            fig.suptitle(
                f"{quantity} - Convergence Optimization Comparison\nTime saved: {time_saved:.2f}s ({(time_saved / original_duration) * 100:.1f}% reduction)\nCompressed duration: {compressed_duration:.2f}s",
                fontsize=14)

            # Set labels
            ax1.set_ylabel(quantity)
            ax2.set_ylabel(quantity)
            ax3.set_ylabel(quantity)
            ax3.set_xlabel('Time [s]')

            plt.tight_layout()

            # Save figure
            plt.savefig(f"./plots/{output_prefix}_{quantity.lower()}.png", dpi=300, bbox_inches='tight')
            print(f"Compressed timeline plot saved to {output_prefix}_{quantity.lower()}.png")

            plt.close(fig)

    def save_optimized_data(self, optimized_data, folder="./optimized_data"):
        """
        Save optimized data to CSV files

        Parameters:
        -----------
        optimized_data : dict
            Dictionary mapping quantity names to optimized (times, values) tuples
        folder : str
            Folder to save the CSV files
        """
        # Create output directory if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        for quantity, (times, values) in optimized_data.items():
            # Create pandas DataFrame
            df = pd.DataFrame({
                'time': times,
                'value': values
            })

            # Remove rows with None values
            df = df.dropna()

            # Save to CSV
            filename = f"{folder}/{quantity.lower()}_optimized.csv"
            df.to_csv(filename, index=False)
            print(f"Optimized data saved to {filename}")

    def print_phase_values(self, data_dict, all_transitions, actual_task_ranges, tasks):
        """
        Print the initial and final values for each quantity in each phase.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        tasks : list
            Original task configurations
        """
        print("\n" + "=" * 80)
        print("PHASE VALUE SUMMARY (ORIGINAL DATA)")
        print("=" * 80)

        # Get ordered list of tasks
        ordered_tasks = []
        for task in tasks:
            ordered_tasks.append(task["name"])

        # Process each task
        for task_name in ordered_tasks:
            # Skip tasks without actual time ranges
            if task_name not in actual_task_ranges:
                continue

            task_start = actual_task_ranges[task_name]["start"]
            task_end = actual_task_ranges[task_name]["end"]

            print(f"\n➤ Task: {task_name} (from t={task_start:.2f}s to t={task_end:.2f}s)")

            # Process each quantity
            for quantity, (times, values) in data_dict.items():
                # Find values at start and end of task
                start_mask = (times >= task_start) & (times <= task_start + 0.1)  # Small window at start
                end_mask = (times >= task_end - 0.1) & (times <= task_end)  # Small window at end

                # Find the convergence time for this quantity in this task
                convergence = next((t for t in all_transitions[quantity] if
                                    t["type"] == "converge" and t["task"] == task_name), None)

                if np.any(start_mask) and np.any(end_mask):
                    start_idx = np.where(start_mask)[0][0]  # First point in the start window
                    end_idx = np.where(end_mask)[0][-1]  # Last point in the end window

                    start_value = values[start_idx]
                    end_value = values[end_idx]

                    # Calculate percent change
                    if abs(start_value) > 1e-10:  # Avoid division by very small numbers
                        percent_change = ((end_value - start_value) / abs(start_value)) * 100
                    else:
                        percent_change = 0

                    # Format the output with color coding for better readability
                    if abs(percent_change) < 0.1:
                        status = "stable"
                    elif abs(percent_change) < 1.0:
                        status = "minor change"
                    else:
                        status = "significant change"

                    # Print with convergence information if available
                    if convergence:
                        conv_time = convergence["time"]
                        conv_mask = (times >= conv_time - 0.1) & (times <= conv_time + 0.1)

                        if np.any(conv_mask):
                            conv_idx = np.where(conv_mask)[0][0]
                            conv_value = values[conv_idx]

                            print(f"  • {quantity}: {start_value:.4g} → {end_value:.4g} ({percent_change:.2f}% change, {status})")
                            print(f"    ◦ Converged at t={conv_time:.2f}s with value {conv_value:.4g}")
                    else:
                        print(f"  • {quantity}: {start_value:.4g} → {end_value:.4g} ({percent_change:.2f}% change, {status})")
                else:
                    print(f"  • {quantity}: No data available for this phase")

    def print_optimized_phase_values(self, data_dict, optimized_data, all_transitions, actual_task_ranges, tasks):
        """
        Print the initial and final values for each quantity in each phase using optimized data.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to original (times, values) tuples
        optimized_data : dict
            Dictionary mapping quantity names to optimized (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        tasks : list
            Original task configurations
        """
        print("\n" + "=" * 80)
        print("PHASE VALUE SUMMARY (OPTIMIZED DATA)")
        print("=" * 80)

        # Get ordered list of tasks
        ordered_tasks = []
        for task in tasks:
            ordered_tasks.append(task["name"])

        # Process each task
        for task_name in ordered_tasks:
            # Skip tasks without actual time ranges
            if task_name not in actual_task_ranges:
                continue

            task_start = actual_task_ranges[task_name]["start"]
            task_end = actual_task_ranges[task_name]["end"]

            print(f"\n➤ Task: {task_name} (from t={task_start:.2f}s to t={task_end:.2f}s)")

            # Process each quantity
            for quantity, (times, values) in data_dict.items():
                # Get optimized data
                opt_times, opt_values = optimized_data[quantity]

                # Find values at start and end of task in optimized data
                start_mask = (opt_times >= task_start) & (opt_times <= task_start + 0.1)  # Small window at start
                end_mask = (opt_times >= task_end - 0.1) & (opt_times <= task_end)  # Small window at end

                # Find the convergence time for this quantity in this task
                convergence = next((t for t in all_transitions[quantity] if
                                    t["type"] == "converge" and t["task"] == task_name), None)

                # Check if this task is included in the optimized data
                if np.any(start_mask) and np.any(end_mask):
                    # Get the indices in the optimized data
                    start_idx = np.where(start_mask)[0][0]  # First point in the start window
                    end_idx = np.where(end_mask)[0][-1]  # Last point in the end window

                    # Make sure the values aren't None
                    if opt_values[start_idx] is not None and opt_values[end_idx] is not None:
                        start_value = opt_values[start_idx]
                        end_value = opt_values[end_idx]

                        # Calculate percent change
                        if abs(start_value) > 1e-10:  # Avoid division by very small numbers
                            percent_change = ((end_value - start_value) / abs(start_value)) * 100
                        else:
                            percent_change = 0

                        # Format the output with color coding for better readability
                        if abs(percent_change) < 0.1:
                            status = "stable"
                        elif abs(percent_change) < 1.0:
                            status = "minor change"
                        else:
                            status = "significant change"

                        # Print with convergence information if available
                        if convergence:
                            conv_time = convergence["time"]
                            # Check if the convergence time is in the optimized data
                            conv_mask = (opt_times >= conv_time - 0.1) & (opt_times <= conv_time + 0.1)

                            if np.any(conv_mask):
                                conv_idx = np.where(conv_mask)[0][0]

                                # Make sure the value isn't None
                                if opt_values[conv_idx] is not None:
                                    conv_value = opt_values[conv_idx]
                                    print(f"  • {quantity}: {start_value:.4g} → {end_value:.4g} ({percent_change:.2f}% change, {status})")
                                    print(f"    ◦ Converged at t={conv_time:.2f}s with value {conv_value:.4g}")
                                else:
                                    print(f"  • {quantity}: {start_value:.4g} → {end_value:.4g} ({percent_change:.2f}% change, {status})")
                            else:
                                print(f"  • {quantity}: {start_value:.4g} → {end_value:.4g} ({percent_change:.2f}% change, {status})")
                        else:
                            print(f"  • {quantity}: {start_value:.4g} → {end_value:.4g} ({percent_change:.2f}% change, {status})")
                    else:
                        print(f"  • {quantity}: Data contains None values at phase boundaries")
                else:
                    # Check if this phase is completely skipped in the optimized data
                    if convergence:
                        print(f"  • {quantity}: Phase skipped after convergence at t={convergence['time']:.2f}s")
                    else:
                        print(f"  • {quantity}: No optimized data available for this phase")

    def generate_summary_table(self, data_dict, optimized_data, all_transitions, actual_task_ranges, tasks, auto_save=True, filename="phase_value_summary.csv"):
        """
        Generate a summary table of all phases, quantities, and their values.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping quantity names to original (times, values) tuples
        optimized_data : dict
            Dictionary mapping quantity names to optimized (times, values) tuples
        all_transitions : dict
            Dictionary mapping quantity names to transitions lists
        actual_task_ranges : dict
            Dictionary mapping task names to their actual time ranges
        tasks : list
            Original task configurations
        auto_save : bool, optional
            Whether to automatically save the table to CSV without prompting
        filename : str, optional
            Name of the CSV file to create if auto_save is True
        """
        # Get ordered list of tasks
        ordered_tasks = []

        # Create a map of task names to their original max times
        task_max_times = {}
        for task in tasks:
            if task["name"] not in ["stopJob"]:  # Exclude end tasks with no convergence criteria
                ordered_tasks.append(task["name"])
                # Store the original max time for each task
                if "max_time" in task:
                    task_max_times[task["name"]] = task["max_time"]

        # Create summary table data structure
        summary_table = {}

        # Process each quantity
        for quantity, (times, values) in data_dict.items():
            summary_table[quantity] = {}

            # Get optimized data
            opt_times, opt_values = optimized_data[quantity]

            # Find the previous task end time to calculate cumulative start times
            prev_end_time = 0.0

            # Process each task
            for task_index, task_name in enumerate(ordered_tasks):
                # Skip tasks without actual time ranges
                if task_name not in actual_task_ranges:
                    continue

                task_start = actual_task_ranges[task_name]["start"]
                task_end = actual_task_ranges[task_name]["end"]

                # Calculate the original max end time (non-converged end time)
                # Use either the stored max time from the task definition or the next task's start
                original_max_end_time = task_start + task_max_times.get(task_name, 0)

                # Find values at start and end of task in original data
                start_mask = (times >= task_start) & (times <= task_start + 0.1)
                end_mask = (times >= task_end - 0.1) & (times <= task_end)

                # Find the convergence time for this quantity in this task
                convergence = next((t for t in all_transitions[quantity] if
                                    t["type"] == "converge" and t["task"] == task_name), None)

                # Initialize task data
                task_data = {
                    "start_time": task_start,
                    "end_time": task_end,
                    "original_start_value": None,
                    "original_end_value": None,
                    "optimized_start_value": None,
                    "optimized_end_value": None,
                    "converged": convergence is not None,
                    "convergence_time": None,
                    "convergence_value": None,
                    "time_saved": 0
                }

                # Get original values
                if np.any(start_mask) and np.any(end_mask):
                    start_idx = np.where(start_mask)[0][0]
                    end_idx = np.where(end_mask)[0][-1]

                    task_data["original_start_value"] = values[start_idx]
                    task_data["original_end_value"] = values[end_idx]

                # Get optimized values
                opt_start_mask = (opt_times >= task_start) & (opt_times <= task_start + 0.1)
                opt_end_mask = (opt_times >= task_end - 0.1) & (opt_times <= task_end)

                if np.any(opt_start_mask) and np.any(opt_end_mask):
                    opt_start_idx = np.where(opt_start_mask)[0][0]
                    opt_end_idx = np.where(opt_end_mask)[0][-1]

                    if opt_values[opt_start_idx] is not None and opt_values[opt_end_idx] is not None:
                        task_data["optimized_start_value"] = opt_values[opt_start_idx]
                        task_data["optimized_end_value"] = opt_values[opt_end_idx]

                # Get convergence data and calculate true time saved
                if convergence:
                    task_data["convergence_time"] = convergence["time"]

                    # Find value at convergence time
                    conv_mask = (times >= convergence["time"] - 0.1) & (times <= convergence["time"] + 0.1)
                    if np.any(conv_mask):
                        conv_idx = np.where(conv_mask)[0][0]
                        task_data["convergence_value"] = values[conv_idx]

                    # Calculate time saved - use the original max end time for correct calculation
                    if task_data["convergence_time"] < original_max_end_time:
                        task_data["time_saved"] = original_max_end_time - task_data["convergence_time"]

                        # For WFT in propulsionVariation - calculate the special case
                        if quantity == "WFT" and task_name == "propulsionVariation":
                            # The time saved should be the time between convergence and the end of the phase
                            # If WFT converges before Fx, but we had to wait for Fx anyway
                            fx_convergence = next((t for t in all_transitions["Fx"] if
                                                   t["type"] == "converge" and t["task"] == task_name), None)
                            if fx_convergence and fx_convergence["time"] > task_data["convergence_time"]:
                                # Only count time saved if we're not waiting for another required quantity
                                task_data["time_saved"] = 0  # Reset if we had to wait for Fx anyway

                # Add task data to summary
                summary_table[quantity][task_name] = task_data

        # Fix the time saved for quantities in phases where multiple quantities need to converge
        # For example, in propulsionVariation, both Fx and WFT need to converge
        for task_name in ordered_tasks:
            if task_name in ["propulsionVariation", "final"]:  # Phases that require multiple convergences
                # Find the latest convergence time among required quantities
                latest_convergence_time = 0
                for quantity in required_convergence.get(task_name, []):
                    if quantity in summary_table and task_name in summary_table[quantity]:
                        if summary_table[quantity][task_name]["converged"]:
                            conv_time = summary_table[quantity][task_name]["convergence_time"]
                            if conv_time > latest_convergence_time:
                                latest_convergence_time = conv_time

                # Adjust time saved for early-converging quantities
                for quantity in required_convergence.get(task_name, []):
                    if quantity in summary_table and task_name in summary_table[quantity]:
                        if summary_table[quantity][task_name]["converged"]:
                            conv_time = summary_table[quantity][task_name]["convergence_time"]
                            if conv_time < latest_convergence_time:
                                # This quantity converged early but had to wait for others
                                summary_table[quantity][task_name]["time_saved"] = 0

        # Print summary table
        print("\n" + "=" * 100)
        print("COMPREHENSIVE VALUE SUMMARY (ALL PHASES AND QUANTITIES)")
        print("=" * 100)

        # Print table headers
        headers = ["Quantity", "Phase", "Start Time", "End Time", "Original Start", "Original End", "Optimized Start", "Optimized End",
                   "Converged", "Conv Time", "Conv Value", "Time Saved"]
        print(f"{headers[0]:<10} {headers[1]:<20} {headers[2]:<10} {headers[3]:<10} {headers[4]:<15} {headers[5]:<15} "
              f"{headers[6]:<15} {headers[7]:<15} {headers[8]:<10} {headers[9]:<10} {headers[10]:<15} {headers[11]:<10}")
        print("-" * 150)

        # Print table rows
        for quantity, tasks_data in summary_table.items():
            for task_name, task_data in tasks_data.items():
                # Format values for printing
                orig_start = f"{task_data['original_start_value']:.4g}" if task_data['original_start_value'] is not None else "N/A"
                orig_end = f"{task_data['original_end_value']:.4g}" if task_data['original_end_value'] is not None else "N/A"
                opt_start = f"{task_data['optimized_start_value']:.4g}" if task_data['optimized_start_value'] is not None else "N/A"
                opt_end = f"{task_data['optimized_end_value']:.4g}" if task_data['optimized_end_value'] is not None else "N/A"
                conv = "Yes" if task_data['converged'] else "No"
                conv_time = f"{task_data['convergence_time']:.2f}s" if task_data['convergence_time'] is not None else "N/A"
                conv_val = f"{task_data['convergence_value']:.4g}" if task_data['convergence_value'] is not None else "N/A"
                time_saved = f"{task_data['time_saved']:.2f}s" if task_data['time_saved'] > 0 else "0.00s"

                print(f"{quantity:<10} {task_name:<20} {task_data['start_time']:<10.2f} {task_data['end_time']:<10.2f} "
                      f"{orig_start:<15} {orig_end:<15} {opt_start:<15} {opt_end:<15} "
                      f"{conv:<10} {conv_time:<10} {conv_val:<15} {time_saved:<10}")

        # Auto-save to CSV if enabled
        if auto_save:
            self.export_summary_to_csv(summary_table, filename)

        return summary_table  # Return the summary table for potential further use

    def export_summary_to_csv(self, summary_table, filename="phase_value_summary.csv"):
        """
        Export the summary table to a CSV file.

        Parameters:
        -----------
        summary_table : dict
            Dictionary containing all the summary data
        filename : str
            Name of the CSV file to create
        """
        # Create the CSV data
        csv_data = [["Quantity", "Phase", "Start Time", "End Time", "Original Start", "Original End",
                     "Optimized Start", "Optimized End", "Converged", "Convergence Time",
                     "Convergence Value", "Time Saved"]]

        for quantity, tasks_data in summary_table.items():
            for task_name, task_data in tasks_data.items():
                # Format values
                orig_start = task_data['original_start_value'] if task_data['original_start_value'] is not None else "N/A"
                orig_end = task_data['original_end_value'] if task_data['original_end_value'] is not None else "N/A"
                opt_start = task_data['optimized_start_value'] if task_data['optimized_start_value'] is not None else "N/A"
                opt_end = task_data['optimized_end_value'] if task_data['optimized_end_value'] is not None else "N/A"
                conv = "Yes" if task_data['converged'] else "No"
                conv_time = task_data['convergence_time'] if task_data['convergence_time'] is not None else "N/A"
                conv_val = task_data['convergence_value'] if task_data['convergence_value'] is not None else "N/A"

                csv_data.append([
                    quantity, task_name, task_data['start_time'], task_data['end_time'],
                    orig_start, orig_end, opt_start, opt_end,
                    conv, conv_time, conv_val, task_data['time_saved']
                ])

        # Write to CSV
        import csv
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(csv_data)

        print(f"Summary exported to {filename}")


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
        all_transitions, actual_task_ranges = analyzer.analyze_convergence_multi_quantity(
            data_dict,
            tasks,
            required_convergence
        )

        # Print summary of convergence
        analyzer.print_multi_quantity_summary(all_transitions, actual_task_ranges)

        # Print phase values for original data
        analyzer.print_phase_values(data_dict, all_transitions, actual_task_ranges, tasks)

        # Plot results for each quantity
        analyzer.plot_multi_quantity_results(data_dict, all_transitions, actual_task_ranges, "convergence_with_oscillation")

        # Plot detailed oscillation detection visualizations
        analyzer.plot_oscillation_comparison(data_dict, all_transitions, "largeTimeStep")
        analyzer.plot_oscillation_comparison(data_dict, all_transitions, "propulsionVariation")
        analyzer.plot_oscillation_comparison(data_dict, all_transitions, "final")

        # Create optimized data sets that jump from convergence to next phase
        optimized_data = analyzer.create_optimized_data(data_dict, all_transitions, actual_task_ranges, tasks)

        # Print phase values for optimized data
        analyzer.print_optimized_phase_values(data_dict, optimized_data, all_transitions, actual_task_ranges, tasks)

        # Generate comprehensive summary table and auto-save to CSV
        analyzer.generate_summary_table(data_dict, optimized_data, all_transitions, actual_task_ranges, tasks, auto_save=True)

        # Plot comparison with improved visualization
        analyzer.plot_optimized_comparison(data_dict, optimized_data, all_transitions, actual_task_ranges)

        # Create compressed timeline data
        compressed_data = analyzer.create_compressed_timeline_data(data_dict, optimized_data)

        # Plot compressed data comparison - FIX: all_transitions instead of all_translations
        analyzer.plot_compressed_data(data_dict, optimized_data, compressed_data, all_transitions, actual_task_ranges)

        # Save optimized data to CSV files
        analyzer.save_optimized_data(optimized_data)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Falling back to synthetic data for demonstration...")