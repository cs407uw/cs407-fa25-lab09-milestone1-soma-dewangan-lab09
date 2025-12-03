import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# DATA LOADING AND PROCESSING
def inspect_csv_file(filename, num_lines=5):
    try:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i < num_lines:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break
    except FileNotFoundError:
        print(f"File not found: {filename}")


def load_csv_with_fallbacks(filename):
    strategies = [
        lambda: pd.read_csv(filename, index_col=False),
        lambda: pd.read_csv(filename, on_bad_lines='skip'),
        lambda: pd.read_csv(filename, error_bad_lines=False),
        lambda: pd.read_csv(filename, header=None, on_bad_lines='skip')
    ]

    for idx, strategy in enumerate(strategies):
        try:
            df = strategy()
            print(f"Successfully loaded {filename} with strategy {idx+1}")

            if idx == 3:
                df.columns = df.iloc[0]
                df = df.drop(0).reset_index(drop=True)

            return df
        except Exception as e:
            if idx == len(strategies) - 1:
                print(f"All loading strategies failed: {e}")
    return None


def apply_smoothing(data, method='savgol', window_length=11, polyorder=3, sigma=2):
    if method == 'savgol':
        if window_length >= len(data):
            window_length = len(data) - 1 if len(data) % 2 == 0 else len(data) - 2
        if window_length % 2 == 0:
            window_length -= 1
        return savgol_filter(data, window_length, polyorder)
    elif method == 'gaussian':
        return gaussian_filter1d(data, sigma)
    return data


# PART 1: SENSOR DATA ERROR ANALYSIS

def analyze_sensor_errors(filename='ACCELERATION.csv', save_plots=False):

    df = load_csv_with_fallbacks(filename)
    if df is None:
        print("Failed to load data.")
        inspect_csv_file(filename)
        return None

    # Extract and clean data
    timestamp = df.iloc[:, 0].values
    clean_accel = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    noisy_accel = pd.to_numeric(df.iloc[:, 2], errors='coerce').values

    # Remove NaN values
    valid_mask = ~(np.isnan(clean_accel) | np.isnan(noisy_accel))
    timestamp = timestamp[valid_mask]
    clean_accel = clean_accel[valid_mask]
    noisy_accel = noisy_accel[valid_mask]

    # Integration parameters
    dt = 0.1

    # Integrate acceleration to velocity
    clean_velocity = np.cumsum(clean_accel) * dt
    noisy_velocity = np.cumsum(noisy_accel) * dt

    # Integrate velocity to distance
    clean_distance = np.cumsum(clean_velocity) * dt
    noisy_distance = np.cumsum(noisy_velocity) * dt

    # Create visualization
    create_acceleration_plot(timestamp, clean_accel, noisy_accel, save_plots)
    create_velocity_plot(timestamp, clean_velocity, noisy_velocity, save_plots)
    create_distance_plot(timestamp, clean_distance, noisy_distance, save_plots)

    # Calculate final results
    final_clean = clean_distance[-1]
    final_noisy = noisy_distance[-1]
    diff = abs(final_clean - final_noisy)
    percent_error = (diff / abs(final_clean)) * 100 if final_clean != 0 else 0

    print("PART 1 RESULTS: Sensor Error Analysis")
    print(f"Final distance (Clean): {final_clean:.4f} m")
    print(f"Final distance (Noisy): {final_noisy:.4f} m")
    print(f"Absolute difference: {diff:.4f} m")
    print(f"Percentage error: {percent_error:.2f}%")

    return {'clean': final_clean, 'noisy': final_noisy, 'diff': diff}


def create_acceleration_plot(time, clean, noisy, save_flag):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, clean, 'b-', linewidth=2, label='Clean Acceleration')
    ax.plot(time, noisy, 'r--', linewidth=1.5, label='Noisy Acceleration', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title('Acceleration: Clean vs Noisy', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part1_acceleration.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_velocity_plot(time, clean, noisy, save_flag):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, clean, 'b-', linewidth=2, label='Velocity (Clean)')
    ax.plot(time, noisy, 'r--', linewidth=1.5, label='Velocity (Noisy)', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity Integration Results', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part1_velocity.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_distance_plot(time, clean, noisy, save_flag):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, clean, 'b-', linewidth=2, label='Distance (Clean)')
    ax.plot(time, noisy, 'r--', linewidth=1.5, label='Distance (Noisy)', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    ax.set_title('Distance Integration Results', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part1_distance.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 2: STEP DETECTION

def identify_steps(accel_magnitude, timestamps, threshold=11.5, min_interval=0.2):
    steps = []
    last_step_time = -np.inf
    time_seconds = (timestamps - timestamps[0]) / 1e9

    for i in range(1, len(accel_magnitude) - 1):
        is_peak = (
                accel_magnitude[i] > accel_magnitude[i-1] and
                accel_magnitude[i] > accel_magnitude[i+1] and
                accel_magnitude[i] > threshold
        )

        if is_peak and (time_seconds[i] - last_step_time >= min_interval):
            steps.append(i)
            last_step_time = time_seconds[i]

    return np.array(steps), len(steps)


def detect_steps_in_data(filename='WALKING.csv', show_output=True, use_fixed_threshold=True, save_plots=False):

    df = load_csv_with_fallbacks(filename)
    if df is None:
        print("Failed to load walking data.")
        inspect_csv_file(filename)
        return None, 0

    # Identify relevant columns
    col_names = [str(col).lower() for col in df.columns]

    x_idx = next((i for i, name in enumerate(col_names)
                  if 'accel' in name and 'x' in name), None)
    y_idx = next((i for i, name in enumerate(col_names)
                  if 'accel' in name and 'y' in name), None)
    z_idx = next((i for i, name in enumerate(col_names)
                  if 'accel' in name and 'z' in name), None)
    time_idx = next((i for i, name in enumerate(col_names)
                     if 'time' in name), 0)

    if None in [x_idx, y_idx, z_idx]:
        print("Error: Missing accelerometer data columns")
        return None, 0

    # Extract and clean data
    accel_x = pd.to_numeric(df.iloc[:, x_idx], errors='coerce').values
    accel_y = pd.to_numeric(df.iloc[:, y_idx], errors='coerce').values
    accel_z = pd.to_numeric(df.iloc[:, z_idx], errors='coerce').values
    timestamps = pd.to_numeric(df.iloc[:, time_idx], errors='coerce').values

    # Remove invalid entries
    valid_mask = ~(np.isnan(accel_x) | np.isnan(accel_y) | np.isnan(accel_z) | np.isnan(timestamps))
    accel_x = accel_x[valid_mask]
    accel_y = accel_y[valid_mask]
    accel_z = accel_z[valid_mask]
    timestamps = timestamps[valid_mask]

    # Calculate magnitude and smooth
    magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    time_sec = (timestamps - timestamps[0]) / 1e9
    smoothed = apply_smoothing(magnitude, method='savgol', window_length=21, polyorder=3)

    # Determine threshold
    if use_fixed_threshold:
        threshold = 11.5
    else:
        mean_val = np.mean(smoothed)
        std_val = np.std(smoothed)
        threshold = mean_val + 0.01 * std_val

    # Detect steps
    step_indices, step_count = identify_steps(smoothed, timestamps, threshold=threshold, min_interval=0.2)

    # Visualizations
    if show_output:
        plot_raw_vs_smoothed(time_sec, magnitude, smoothed, save_plots)
        plot_step_detection(time_sec, smoothed, step_indices, step_count, threshold, save_plots)

        print("PART 2 RESULTS: Step Detection")
        print(f"Detected steps: {step_count}")
        print(f"Expected steps: 37")
        print(f"Detection threshold: {threshold:.2f} m/s²")

    return step_indices, step_count


def plot_raw_vs_smoothed(time, raw, smoothed, save_flag):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, raw, 'lightblue', alpha=0.6,
            label='Raw Acceleration Magnitude')
    ax.plot(time, smoothed, 'b-', linewidth=2,
            label='Smoothed (Savitzky-Golay)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title('Acceleration Smoothing for Step Detection',
                 fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part2_smoothing.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_step_detection(time, data, steps, count, threshold, save_flag):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, data, 'b-', linewidth=1.5, label='Smoothed Acceleration')

    if len(steps) > 0:
        ax.plot(time[steps], data[steps], 'ro', markersize=8,
                label=f'Detected Steps ({count})', alpha=0.8)

    ax.axhline(y=threshold, color='g', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold:.1f} m/s²)', alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title(f'Step Detection: {count} Steps Found (Expected: 37)',
                 fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part2_step_detection.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 3: DIRECTION DETECTION

def find_rotation_events(gyro_z, timestamps, peak_thresh=2.5,
                         min_time_gap=1.5, angle_rounding=90):

    time_vals = (timestamps - timestamps[0]) / 1e9
    dt_vals = np.diff(time_vals)
    dt_vals = np.append(dt_vals, dt_vals[-1])

    # Cumulative angle integration
    cumulative_angle = np.cumsum(gyro_z * dt_vals) * (180 / np.pi)

    events = []
    last_event_time = -min_time_gap

    for i in range(1, len(gyro_z) - 1):
        current_time = time_vals[i]

        # Enforce minimum time between events
        if current_time - last_event_time < min_time_gap:
            continue

        # Detect clockwise peak
        if (gyro_z[i] > gyro_z[i-1] and
                gyro_z[i] > gyro_z[i+1] and
                gyro_z[i] > peak_thresh):

            start_idx = max(0, i - 50)
            end_idx = min(len(gyro_z), i + 50)
            angle_change = np.sum(gyro_z[start_idx:end_idx] * dt_vals[start_idx:end_idx]) * (180 / np.pi)

            rounded_angle = round(angle_change / angle_rounding) * angle_rounding

            events.append({
                'index': i,
                'time': current_time,
                'angle': rounded_angle,
                'direction': 'counterclockwise',
                'peak': gyro_z[i],
                'raw_angle': angle_change
            })
            last_event_time = current_time

        # Detect counterclockwise peak
        elif (gyro_z[i] < gyro_z[i-1] and
              gyro_z[i] < gyro_z[i+1] and
              gyro_z[i] < -peak_thresh):

            start_idx = max(0, i - 50)
            end_idx = min(len(gyro_z), i + 50)
            angle_change = np.sum(gyro_z[start_idx:end_idx] *
                                  dt_vals[start_idx:end_idx]) * (180 / np.pi)

            rounded_angle = round(angle_change / angle_rounding) * angle_rounding

            events.append({
                'index': i,
                'time': current_time,
                'angle': rounded_angle,
                'direction': 'clockwise',
                'peak': gyro_z[i],
                'raw_angle': angle_change
            })
            last_event_time = current_time

    return events, cumulative_angle


def analyze_turning_data(filename='TURNING.csv', angle_multiple=90,
                         show_output=True, peak_thresh=2.5, save_plots=False):
    df = load_csv_with_fallbacks(filename)
    if df is None:
        print("Failed to load turning data.")
        inspect_csv_file(filename)
        return None

    # Identify gyroscope column
    col_names = [str(col).lower() for col in df.columns]
    gyro_idx = next((i for i, name in enumerate(col_names)
                     if 'gyro' in name and 'z' in name), None)
    time_idx = next((i for i, name in enumerate(col_names)
                     if 'time' in name), 0)

    if gyro_idx is None:
        print("Error: Missing gyroscope Z-axis data")
        return None

    # Extract and clean data
    gyro_z = pd.to_numeric(df.iloc[:, gyro_idx], errors='coerce').values
    timestamps = pd.to_numeric(df.iloc[:, time_idx], errors='coerce').values

    valid_mask = ~(np.isnan(gyro_z) | np.isnan(timestamps))
    gyro_z = gyro_z[valid_mask]
    timestamps = timestamps[valid_mask]

    time_sec = (timestamps - timestamps[0]) / 1e9

    # Smooth gyroscope data
    smoothed_gyro = apply_smoothing(gyro_z, method='gaussian', sigma=5)

    # Detect turning events
    turns, cumulative_angle = find_rotation_events(
        smoothed_gyro, timestamps, peak_thresh, 1.5, angle_multiple
    )

    # Visualizations
    if show_output:
        plot_gyro_smoothing(time_sec, gyro_z, smoothed_gyro, peak_thresh, save_plots)
        plot_turn_detection(time_sec, smoothed_gyro, turns, peak_thresh, save_plots)
        plot_cumulative_angle(time_sec, cumulative_angle, turns, angle_multiple, save_plots)

        print_results(turns, angle_multiple, peak_thresh)

    return turns


def plot_gyro_smoothing(time, raw, smoothed, threshold, save_flag):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, raw, 'lightcoral', alpha=0.6, label='Raw Gyro Z')
    ax.plot(time, smoothed, 'r-', linewidth=2, label='Smoothed (Gaussian)')
    ax.axhline(y=threshold, color='green', linestyle='--', alpha=0.7,
               label=f'Threshold (±{threshold})')
    ax.axhline(y=-threshold, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax.set_title('Gyroscope Data Smoothing', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part3_smoothing.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_turn_detection(time, data, turns, threshold, save_flag):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, data, 'b-', linewidth=1.5, label='Smoothed Gyro Z')

    for idx, turn in enumerate(turns, 1):
        color = 'darkorange' if turn['direction'] == 'counterclockwise' else 'darkviolet'
        marker = '^' if turn['direction'] == 'counterclockwise' else 'v'
        ax.plot(turn['time'], turn['peak'], marker=marker, color=color,
                markersize=10, zorder=5, label=f'Turn {idx}' if idx == 1 else '')

        offset = 12 if turn['peak'] > 0 else -20
        ax.annotate(f"{idx}", xy=(turn['time'], turn['peak']),
                    xytext=(0, offset), textcoords='offset points',
                    fontsize=10, weight='bold', ha='center', color=color)

    ax.axhline(y=threshold, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-threshold, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax.set_title(f'Turn Detection: {len(turns)} Turns Found (Expected: 8)',
                 fontsize=14, weight='bold')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', lw=2, label='Smoothed Gyro Z'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='darkorange',
               markersize=10, label='CCW Turn'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='darkviolet',
               markersize=10, label='CW Turn'),
        Line2D([0], [0], color='green', linestyle='--', label=f'Threshold (±{threshold})')
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part3_turn_detection.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_cumulative_angle(time, angles, turns, angle_multiple, save_flag):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(time, angles, 'b-', linewidth=2, label='Cumulative Angle')

    for idx, turn in enumerate(turns, 1):
        marker = '^' if turn['direction'] == 'counterclockwise' else 'v'
        color = 'darkorange' if turn['direction'] == 'counterclockwise' else 'darkviolet'
        ax.plot(turn['time'], angles[turn['index']],
                marker=marker, color=color, markersize=10, zorder=5)

        ax.annotate(f"{turn['angle']}°", xy=(turn['time'], angles[turn['index']]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, weight='bold', ha='left', color=color)

    # Add angle grid lines
    for angle in range(-360, 360, angle_multiple):
        if angle != 0:
            ax.axhline(y=angle, color='gray', linestyle=':', alpha=0.2)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cumulative Angle (degrees)', fontsize=12)
    ax.set_title(f'Angle Integration: {len(turns)} Turns Detected',
                 fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_flag:
        plt.savefig('part3_cumulative_angle.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_results(turns, angle_multiple, threshold):
    print("PART 3 RESULTS: Turning Analysis")
    print(f"Detected turns: {len(turns)}")
    print(f"Expected turns: 8 (4 CW + 4 CCW)")
    print(f"Smoothing: Gaussian filter (σ=5)")
    print(f"Peak threshold: ±{threshold} rad/s")
    print(f"Angle rounding: {angle_multiple}° increments")

    cw_count = sum(1 for t in turns if t['direction'] == 'clockwise')
    ccw_count = sum(1 for t in turns if t['direction'] == 'counterclockwise')
    print(f"Clockwise turns: {cw_count}")
    print(f"Counterclockwise turns: {ccw_count}")

    print("\nDetailed turn information:")
    for i, turn in enumerate(turns, 1):
        dir_str = turn['direction']
        print(f"  Turn {i:2d}: {turn['angle']:>4}° {dir_str:<17} at {turn['time']:>5.2f}s")

    if len(turns) == 8:
        print("\nCorrect number of turns detected!")
    else:
        diff = abs(len(turns) - 8)
        print(f"\nIncorrect count (off by {diff})")


# PART 4: TRAJECTORY RECONSTRUCTION

def reconstruct_trajectory(filename="WALKING_AND_TURNING.csv", save_plots=False):

    # Detect steps and turns
    step_indices, step_count = detect_steps_in_data(
        filename, show_output=False, use_fixed_threshold=False
    )
    turns = analyze_turning_data(
        filename, angle_multiple=45, show_output=False, peak_thresh=2.2
    )

    # Load timestamp data
    df = load_csv_with_fallbacks(filename)
    timestamps = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    time_sec = (timestamps - timestamps[0]) / 1e9

    # Sort turns chronologically
    turns.sort(key=lambda t: t['index'])

    # Initialize tracking variables
    current_heading = 90
    step_size = 1.0
    x_pos, y_pos = 0.0, 0.0
    x_old = [x_pos]
    y_old = [y_pos]
    turn_index = 0

    # Process each step
    for step_idx in step_indices:
        # Apply any turns that occur before this step
        while turn_index < len(turns) and turns[turn_index]['index'] < step_idx:
            current_heading += turns[turn_index]['angle']
            turn_index += 1

        # Move forward
        rad_heading = np.radians(current_heading)
        x_pos += step_size * np.cos(rad_heading)
        y_pos += step_size * np.sin(rad_heading)

        x_old.append(x_pos)
        y_old.append(y_pos)

    # Create trajectory plot
    plot_trajectory(x_old, y_old, turns, time_sec, save_plots)

    # Calculate and display statistics
    total_distance = step_count * step_size
    final_x, final_y = x_old[-1], y_old[-1]
    displacement = np.sqrt(final_x**2 + final_y**2)

    print("PART 4 RESULTS: Trajectory Reconstruction")
    print(f"Total steps: {step_count}")
    print(f"Step length: {step_size} m")
    print(f"Total distance: {total_distance:.2f} m")
    print(f"Final position: ({final_x:.2f}, {final_y:.2f})")
    print(f"Displacement from start: {displacement:.2f} m")
    print(f"Number of turns: {len(turns)}")

    print("\nTurn sequence:")
    for i, turn in enumerate(turns, 1):
        print(f"  Turn {i}: {turn['angle']}° ({turn['direction']}) at {turn['time']:.2f}s")

    return x_old, y_old


def plot_trajectory(x_vals, y_vals, turns, time_data, save_flag):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot path
    ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='Walking Path', alpha=0.8)

    # Mark start and end
    ax.plot(x_vals[0], y_vals[0], 'go', markersize=14,
            label='Start', markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(x_vals[-1], y_vals[-1], 'ro', markersize=14,
            label='End', markeredgecolor='black', markeredgewidth=1.5)

    # Add intermediate markers
    if len(x_vals) > 10:
        for i in range(5, len(x_vals), 5):
            ax.plot(x_vals[i], y_vals[i], 'b.', markersize=8, alpha=0.6)

    # Add north arrow
    ax.arrow(x_vals[0], y_vals[0], 0, 1.5,
             head_width=0.3, head_length=0.4, fc='green', ec='green', zorder=3)
    ax.text(x_vals[0]-0.8, y_vals[0]+2, 'N', fontsize=14, weight='bold', color='green')

    # Mark turn locations
    for i, turn in enumerate(turns, 1):
        # Find closest path point to turn time
        time_idx = min(range(len(time_data)),
                       key=lambda j: abs(time_data[j] - turn['time']))
        if time_idx < len(x_vals):
            ax.plot(x_vals[time_idx], y_vals[time_idx], 's',
                    color='orange', markersize=8, alpha=0.7)
            ax.annotate(f"T{i}: {turn['angle']}°",
                        xy=(x_vals[time_idx], y_vals[time_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, weight='bold', color='darkorange')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Reconstructed Walking Trajectory\n(1m Steps + 45° Turns)',
                 fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    if save_flag:
        plt.savefig('part4_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()


# MAIN EXECUTION
def execute_all_analyses(save_figures=False):

    try:
        sensor_results = analyze_sensor_errors('ACCELERATION.csv', save_figures)
    except Exception as e:
        print(f"Part 1 error: {e}")

    try:
        step_data = detect_steps_in_data('WALKING.csv', save_plots=save_figures)
    except Exception as e:
        print(f"Part 2 error: {e}")

    try:
        turn_data = analyze_turning_data('TURNING.csv', save_plots=save_figures)
    except Exception as e:
        print(f"Part 3 error: {e}")

    try:
        trajectory = reconstruct_trajectory('WALKING_AND_TURNING.csv', save_figures)
    except Exception as e:
        print(f"Part 4 error: {e}")

if __name__ == "__main__":
    execute_all_analyses(save_figures=True)