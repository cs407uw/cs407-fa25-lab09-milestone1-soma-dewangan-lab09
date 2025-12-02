import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# BASIC FUNCTIONS
def inspect_csv(filename, num_lines=5):
    try:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i < num_lines:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break
    except FileNotFoundError:
        print(f"File not found: {filename}")


def load_csv_safe(filename):
    try:
        df = pd.read_csv(filename, index_col=False)
        print(f"Successfully loaded {filename}")
        return df
    except Exception as e:
        print(f"Loading failed: {e}")

    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
        print(df)
        return df
    except:
        pass

    try:
        df = pd.read_csv(filename, error_bad_lines=False)
        return df
    except:
        pass

    try:
        df = pd.read_csv(filename, header=None, on_bad_lines='skip')
        df.columns = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"All loading strategies failed: {e}")
        return None


def smooth_data(data, method='savgol', window_length=11, polyorder=3, sigma=2):
    if method == 'savgol':
        if window_length >= len(data):
            window_length = len(data) - 1 if len(data) % 2 == 0 else len(data) - 2
        if window_length % 2 == 0:
            window_length -= 1
        return savgol_filter(data, window_length, polyorder)
    elif method == 'gaussian':
        return gaussian_filter1d(data, sigma)
    else:
        return data


# PART 1: Understanding Sensor Data Errors

def part1_sensor_errors(filename='ACCELERATION.csv', save_figures=False):

    df = load_csv_safe(filename)
    if df is None:
        print("Failed to load data.")
        inspect_csv(filename)
        return None

    timestamp = df.iloc[:, 0].values
    acceleration = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    noisy_acceleration = pd.to_numeric(df.iloc[:, 2], errors='coerce').values

    valid_mask = ~(np.isnan(acceleration) | np.isnan(noisy_acceleration))
    timestamp = timestamp[valid_mask]
    acceleration = acceleration[valid_mask]
    noisy_acceleration = noisy_acceleration[valid_mask]

    dt = 0.1
    velocity_actual = np.cumsum(acceleration) * dt
    velocity_noisy = np.cumsum(noisy_acceleration) * dt
    distance_actual = np.cumsum(velocity_actual) * dt
    distance_noisy = np.cumsum(velocity_noisy) * dt

    # Plot 1: Acceleration
    plt.figure(figsize=(10, 6))
    plt.plot(timestamp, acceleration, 'b-', linewidth=2, label='Actual Acceleration')
    plt.plot(timestamp, noisy_acceleration, 'r--', linewidth=1.5, label='Noisy Acceleration', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Acceleration (m/s²)', fontsize=12)
    plt.title('Acceleration Comparison: Actual vs Noisy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_figures:
        plt.savefig('part1_acceleration.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Velocity
    plt.figure(figsize=(10, 6))
    plt.plot(timestamp, velocity_actual, 'b-', linewidth=2, label='Velocity from Actual')
    plt.plot(timestamp, velocity_noisy, 'r--', linewidth=1.5, label='Velocity from Noisy', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.title('Velocity Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_figures:
        plt.savefig('part1_velocity.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: Distance
    plt.figure(figsize=(10, 6))
    plt.plot(timestamp, distance_actual, 'b-', linewidth=2, label='Distance from Actual')
    plt.plot(timestamp, distance_noisy, 'r--', linewidth=1.5, label='Distance from Noisy', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Distance (m)', fontsize=12)
    plt.title('Distance Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_figures:
        plt.savefig('part1_distance.png', dpi=300, bbox_inches='tight')
    plt.show()

    final_actual = distance_actual[-1]
    final_noisy = distance_noisy[-1]
    difference = abs(final_actual - final_noisy)

    print(f"\nResults:")
    print(f"Final distance (Actual): {final_actual:.4f} m")
    print(f"Final distance (Noisy): {final_noisy:.4f} m")
    print(f"Difference: {difference:.4f} m")
    print(f"Percentage Error: {(difference/abs(final_actual))*100:.2f}%\n")

    return {'actual': final_actual, 'noisy': final_noisy, 'diff': difference}



# PART 2: Step Detection

def detect_steps(accel_magnitude, timestamps, threshold=11.5, min_step_time=0.2):
    steps = []
    last_step_time = -np.inf

    times = (timestamps - timestamps[0]) / 1e9

    for i in range(1, len(accel_magnitude) - 1):
        if (accel_magnitude[i] > accel_magnitude[i-1] and
                accel_magnitude[i] > accel_magnitude[i+1] and
                accel_magnitude[i] > threshold):

            if times[i] - last_step_time >= min_step_time:
                steps.append(i)
                last_step_time = times[i]

    return np.array(steps), len(steps)


def part2_step_detection(filename='WALKING.csv', output=True, hardcoded=True, save_figures=False):
    df = load_csv_safe(filename)
    if df is None:
        print("Failed to load data.")
        inspect_csv(filename)
        return None, 0

    cols_lower = [str(col).lower() for col in df.columns]
    accel_x_idx = next((i for i, col in enumerate(cols_lower) if 'accel' in col and 'x' in col), None)
    accel_y_idx = next((i for i, col in enumerate(cols_lower) if 'accel' in col and 'y' in col), None)
    accel_z_idx = next((i for i, col in enumerate(cols_lower) if 'accel' in col and 'z' in col), None)
    timestamp_idx = next((i for i, col in enumerate(cols_lower) if 'time' in col), 0)

    if None in [accel_x_idx, accel_y_idx, accel_z_idx]:
        print("Error: Could not find accelerometer columns")
        return None, 0

    accel_x = pd.to_numeric(df.iloc[:, accel_x_idx], errors='coerce').values
    accel_y = pd.to_numeric(df.iloc[:, accel_y_idx], errors='coerce').values
    accel_z = pd.to_numeric(df.iloc[:, accel_z_idx], errors='coerce').values
    timestamps = pd.to_numeric(df.iloc[:, timestamp_idx], errors='coerce').values

    valid_mask = ~(np.isnan(accel_x) | np.isnan(accel_y) | np.isnan(accel_z) | np.isnan(timestamps))
    accel_x = accel_x[valid_mask]
    accel_y = accel_y[valid_mask]
    accel_z = accel_z[valid_mask]
    timestamps = timestamps[valid_mask]

    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    time_seconds = (timestamps - timestamps[0]) / 1e9

    accel_smoothed = smooth_data(accel_magnitude, method='savgol', window_length=21, polyorder=3)

    if output:
        # Plot: Raw vs Smoothed Acceleration Magnitude
        plt.figure(figsize=(12, 6))
        plt.plot(time_seconds, accel_magnitude, 'lightblue', alpha=0.6,
                 label='Raw Acceleration Magnitude')
        plt.plot(time_seconds, accel_smoothed, 'b-', linewidth=2,
                 label='Smoothed (Savitzky-Golay, window=21, polyorder=3)')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
        plt.title('Step Detection: Raw vs Smoothed Acceleration Magnitude',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_figures:
            plt.savefig('part2_smoothing.png', dpi=300, bbox_inches='tight')
        plt.show()

    meanA = np.mean(accel_smoothed)
    stdA = np.std(accel_smoothed)
    threshold = meanA + 0.01 * stdA

    if hardcoded == True:
        threshold = 11.5

    step_indices, step_count = detect_steps(accel_smoothed, timestamps, threshold=threshold, min_step_time=0.2)

    if output:
        # Plot: Step Detection Results
        plt.figure(figsize=(12, 6))
        plt.plot(time_seconds, accel_smoothed, 'b-', linewidth=1.5, label='Smoothed Acceleration')

        if len(step_indices) > 0:
            plt.plot(time_seconds[step_indices], accel_smoothed[step_indices],
                     'ro', markersize=8, label=f'Detected Steps ({step_count})', alpha=0.8)

        plt.axhline(y=threshold, color='g', linestyle='--', linewidth=2,
                    label=f'Threshold ({threshold:.1f} m/s²)', alpha=0.7)

        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
        plt.title(f'Step Detection Results: {step_count} Steps Detected (Expected: 37)',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_figures:
            plt.savefig('part2_step_detection.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nResults:")
        print(f"Total steps detected: {step_count}")
        print(f"Expected steps: 37")
        print(f"Threshold used: {threshold:.2f} m/s²")
        print(f"Smoothing method: Savitzky-Golay filter (window=21, polyorder=3)")
        if step_count == 37:
            print("✓ Correct number of steps detected!")
        else:
            print(f"Detection accuracy: {(min(step_count, 37)/37)*100:.1f}%")
        print()

    return step_indices, step_count



# PART 3: Direction Detection

def detect_turns_peak_based(gyro_z, timestamps, peak_threshold=2.5, min_turn_time=1.5, angle_mult=90):

    times = (timestamps - timestamps[0]) / 1e9
    dt = np.diff(times)
    dt = np.append(dt, dt[-1])

    cumulative_angle = np.cumsum(gyro_z * dt) * (180/np.pi)

    turns = []
    last_turn_time = -min_turn_time

    for i in range(1, len(gyro_z) - 1):
        current_time = times[i]

        if current_time - last_turn_time < min_turn_time:
            continue

        if (gyro_z[i] > gyro_z[i-1] and
                gyro_z[i] > gyro_z[i+1] and
                gyro_z[i] > peak_threshold):

            window_start = max(0, i - 50)
            window_end = min(len(gyro_z), i + 50)
            angle_change = np.sum(gyro_z[window_start:window_end] * dt[window_start:window_end]) * (180/np.pi)

            rounded_angle = round(angle_change / angle_mult) * angle_mult

            turns.append({
                'index': i,
                'time': current_time,
                'angle': rounded_angle,
                'direction': 'counterclockwise',
                'peak_velocity': gyro_z[i],
                'raw_angle': angle_change
            })
            last_turn_time = current_time

        elif (gyro_z[i] < gyro_z[i-1] and
              gyro_z[i] < gyro_z[i+1] and
              gyro_z[i] < -peak_threshold):

            window_start = max(0, i - 50)
            window_end = min(len(gyro_z), i + 50)
            angle_change = np.sum(gyro_z[window_start:window_end] * dt[window_start:window_end]) * (180/np.pi)

            rounded_angle = round(angle_change / angle_mult) * angle_mult

            turns.append({
                'index': i,
                'time': current_time,
                'angle': rounded_angle,
                'direction': 'clockwise',
                'peak_velocity': gyro_z[i],
                'raw_angle': angle_change
            })
            last_turn_time = current_time

    return turns, cumulative_angle


def part3_direction_detection(filename='TURNING.csv', angle_mult=90, output=True, peak_threshold=2.5, save_figures=False):

    df = load_csv_safe(filename)
    if df is None:
        print("Failed to load data.")
        inspect_csv(filename)
        return None

    cols_lower = [str(col).lower() for col in df.columns]
    gyro_z_idx = next((i for i, col in enumerate(cols_lower) if 'gyro' in col and 'z' in col), None)
    timestamp_idx = next((i for i, col in enumerate(cols_lower) if 'time' in col), 0)

    if gyro_z_idx is None:
        print("Error: Could not find gyro_z column")
        return None

    gyro_z = pd.to_numeric(df.iloc[:, gyro_z_idx], errors='coerce').values
    timestamps = pd.to_numeric(df.iloc[:, timestamp_idx], errors='coerce').values

    valid_mask = ~(np.isnan(gyro_z) | np.isnan(timestamps))
    gyro_z = gyro_z[valid_mask]
    timestamps = timestamps[valid_mask]

    time_seconds = (timestamps - timestamps[0]) / 1e9

    # Smooth the data using Gaussian filter
    gyro_z_smoothed = smooth_data(gyro_z, method='gaussian', sigma=5)

    # Plot 1: Raw vs Smoothed Gyroscope Data
    if output:
        plt.figure(figsize=(12, 6))
        plt.plot(time_seconds, gyro_z, 'lightcoral', alpha=0.6, label='Raw Gyroscope Z')
        plt.plot(time_seconds, gyro_z_smoothed, 'r-', linewidth=2,
                 label='Smoothed (Gaussian Filter, σ=5)')
        plt.axhline(y=peak_threshold, color='green', linestyle='--', alpha=0.7,
                    label=f'Peak Threshold (±{peak_threshold} rad/s)')
        plt.axhline(y=-peak_threshold, color='green', linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Angular Velocity (rad/s)', fontsize=12)
        plt.title('Direction Detection: Raw vs Smoothed Gyroscope Z-axis',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_figures:
            plt.savefig('part3_smoothing.png', dpi=300, bbox_inches='tight')
        plt.show()

    turns, cumulative_angle = detect_turns_peak_based(
        gyro_z_smoothed,
        timestamps,
        peak_threshold=peak_threshold,
        min_turn_time=1.5,
        angle_mult=angle_mult
    )

    # Plot 2: Turn Detection Results
    if output:
        plt.figure(figsize=(12, 6))
        plt.plot(time_seconds, gyro_z_smoothed, 'b-', linewidth=1.5, label='Smoothed Gyroscope Z')

        for i, turn in enumerate(turns, 1):
            color = 'darkorange' if turn['direction'] == 'counterclockwise' else 'darkviolet'
            marker = '^' if turn['direction'] == 'counterclockwise' else 'v'
            plt.plot(turn['time'], turn['peak_velocity'], marker=marker, color=color,
                     markersize=10, zorder=5, label=f'Turn {i}' if i == 1 else '')

            offset = 12 if turn['peak_velocity'] > 0 else -20
            plt.annotate(f"{i}", xy=(turn['time'], turn['peak_velocity']),
                         xytext=(0, offset), textcoords='offset points',
                         fontsize=10, fontweight='bold', ha='center', color=color)

        plt.axhline(y=peak_threshold, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=-peak_threshold, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Angular Velocity (rad/s)', fontsize=12)
        plt.title(f'Turn Detection: {len(turns)} Turns Detected (Expected: 8)',
                  fontsize=14, fontweight='bold')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='b', lw=2, label='Smoothed Gyro Z'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='darkorange',
                   markersize=10, label='CCW Turn (^)'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='darkviolet',
                   markersize=10, label='CW Turn (v)'),
            Line2D([0], [0], color='green', linestyle='--', label=f'Threshold (±{peak_threshold})')
        ]
        plt.legend(handles=legend_elements, fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_figures:
            plt.savefig('part3_turn_detection.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot 3: Cumulative Angle
        plt.figure(figsize=(12, 7))
        plt.plot(time_seconds, cumulative_angle, 'b-', linewidth=2, label='Cumulative Angle')

        for i, turn in enumerate(turns, 1):
            marker = '^' if turn['direction'] == 'counterclockwise' else 'v'
            color = 'darkorange' if turn['direction'] == 'counterclockwise' else 'darkviolet'
            plt.plot(turn['time'], cumulative_angle[turn['index']],
                     marker=marker, color=color, markersize=10, zorder=5)

            plt.annotate(f"{turn['angle']}°", xy=(turn['time'], cumulative_angle[turn['index']]),
                         xytext=(10, 5), textcoords='offset points',
                         fontsize=10, fontweight='bold', ha='left', color=color)

        for angle in range(-360, 360, angle_mult):
            if angle != 0:
                plt.axhline(y=angle, color='gray', linestyle=':', alpha=0.2)

        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Cumulative Angle (degrees)', fontsize=12)
        plt.title(f'Cumulative Angle Integration: {len(turns)} Turns Detected',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        if save_figures:
            plt.savefig('part3_cumulative_angle.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nResults:")
        print(f"Total turns detected: {len(turns)}")
        print(f"Expected turns: 8 (4 counterclockwise + 4 clockwise)")
        print(f"Smoothing method: Gaussian filter (σ=5)")
        print(f"Peak threshold: ±{peak_threshold} rad/s")
        print(f"Angle rounding multiple: {angle_mult}°")

        cw_turns = sum(1 for t in turns if t['direction'] == 'clockwise')
        ccw_turns = sum(1 for t in turns if t['direction'] == 'counterclockwise')
        print(f"Counterclockwise turns: {ccw_turns}")
        print(f"Clockwise turns: {cw_turns}")

        print("\nDetected turns:")
        for i, turn in enumerate(turns, 1):
            print(f"  Turn {i}: {turn['angle']:>4}° ({turn['direction']:<17}) at t={turn['time']:>5.2f}s")

        if len(turns) == 8:
            print("\n✓ Correct number of turns detected!")
        else:
            print(f"\n✗ Incorrect number of turns detected (off by {abs(len(turns) - 8)})")

    print()
    return turns


# PART 4: Trajectory Plotting

def part4_trajectory_plotting(filename="WALKING_AND_TURNING.csv", save_figures=False):

    step_indices, step_count = part2_step_detection(filename, output=False, hardcoded=False)
    turns = part3_direction_detection(filename, angle_mult=45, output=False, peak_threshold=2.2)

    df = load_csv_safe(filename)
    timestamps = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    time_sec = (timestamps - timestamps[0]) / 1e9

    turns = sorted(turns, key=lambda t: t['index'])

    heading = 90
    step_length = 1.0
    x, y = 0.0, 0.0
    path_x = [x]
    path_y = [y]
    turn_ptr = 0


    for idx in step_indices:
        while turn_ptr < len(turns) and turns[turn_ptr]['index'] < idx:
            heading += turns[turn_ptr]['angle']
            turn_ptr += 1

        x += step_length * np.cos(np.radians(heading))
        y += step_length * np.sin(np.radians(heading))

        path_x.append(x)
        path_y.append(y)

    # Plot trajectory
    plt.figure(figsize=(10, 10))

    # Plot path
    plt.plot(path_x, path_y, 'b-', linewidth=2.5, label='Walking Path', alpha=0.8)

    # Plot start and end points
    plt.plot(path_x[0], path_y[0], 'go', markersize=14, label='Start',
             markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(path_x[-1], path_y[-1], 'ro', markersize=14, label='End',
             markeredgecolor='black', markeredgewidth=1.5)

    # Plot intermediate steps
    if len(path_x) > 10:
        for i in range(5, len(path_x), 5):
            plt.plot(path_x[i], path_y[i], 'b.', markersize=8, alpha=0.6)

    # Add North arrow and label
    plt.arrow(path_x[0], path_y[0], 0, 1.5,
              head_width=0.3, head_length=0.4, fc='green', ec='green', zorder=3)
    plt.text(path_x[0]-0.8, path_y[0]+2, 'N', fontsize=14, fontweight='bold', color='green')

    # Add turn annotations
    for i, turn in enumerate(turns, 1):
        # Find closest point in path to turn time
        turn_time = turn['time']
        path_idx = min(range(len(time_sec)), key=lambda j: abs(time_sec[j] - turn_time))
        if path_idx < len(path_x):
            plt.plot(path_x[path_idx], path_y[path_idx], 's', color='orange',
                     markersize=8, alpha=0.7)
            plt.annotate(f"T{i}: {turn['angle']}°",
                         xy=(path_x[path_idx], path_y[path_idx]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, fontweight='bold', color='darkorange')

    plt.axis("equal")
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title('Walking Trajectory Reconstruction\n(1m Steps + 45° Turns)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    if save_figures:
        plt.savefig('part4_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

    total_distance = step_count * step_length
    final_x, final_y = path_x[-1], path_y[-1]
    displacement = np.sqrt(final_x**2 + final_y**2)

    print("\nTrajectory Results:")
    print(f"Total steps: {step_count}")
    print(f"Step length: {step_length} m")
    print(f"Total distance traveled: {total_distance:.2f} m")
    print(f"Final position: ({final_x:.2f}, {final_y:.2f})")
    print(f"Straight-line displacement: {displacement:.2f} m")
    print(f"Number of turns: {len(turns)}")

    print("\nTurn details:")
    for i, turn in enumerate(turns, 1):
        print(f"  Turn {i}: {turn['angle']}° ({turn['direction']}) at t={turn['time']:.2f}s")

    return path_x, path_y


# MAIN EXECUTION

def main(save_all_figures=False):
    # Part 1: Sensor Data Errors
    try:
        print("RUNNING PART 1: Sensor Data Errors")
        results1 = part1_sensor_errors('ACCELERATION.csv', save_figures=save_all_figures)
    except Exception as e:
        print(f"Part 1 Error: {e}\n")

    # Part 2: Step Detection
    try:
        print("RUNNING PART 2: Step Detection")
        step_indices, step_count = part2_step_detection('WALKING.csv', save_figures=save_all_figures)
    except Exception as e:
        print(f"Part 2 Error: {e}\n")

    # Part 3: Direction Detection
    try:
        print("RUNNING PART 3: Direction Detection")
        turns = part3_direction_detection('TURNING.csv', save_figures=save_all_figures)
    except Exception as e:
        print(f"Part 3 Error: {e}\n")

    # Part 4: Trajectory Plotting
    try:
        print("RUNNING PART 4: Trajectory Plotting")
        traj_x, traj_y = part4_trajectory_plotting('WALKING_AND_TURNING.csv', save_figures=save_all_figures)
    except Exception as e:
        print(f"Part 4 Error: {e}\n")

if __name__ == "__main__":
    main(save_all_figures=True)