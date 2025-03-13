import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Logging Setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("run_gen.log"), logging.StreamHandler()]
)
log = logging.getLogger()

log.info("Starting data generation...")

# Set up params
np.random.seed(42)  # Fixed seed for testing
num_days = 365
start_date = datetime.now() - timedelta(days=num_days)
# 3 runs per week
num_runs = num_days//7 * 3
log.debug(f"Will generate {num_runs} runs over {num_days} days")

# Generate dates - this might have clustering issues
dates = [start_date + timedelta(days=np.random.randint(0, num_days)) for _ in range(num_runs)]


# Generate distances - normal distribution seems reasonable
distances_run_km = np.random.normal(loc=10, scale=3, size=num_runs).clip(3, 42.2)
log.debug(f"Min/max distances: {min(distances_run_km):.1f}/{max(distances_run_km):.1f}km")

# Pace data
paces_min_per_km = np.random.normal(loc=5.5, scale=0.5, size=num_runs).clip(4, 7)
# Sanity check our pace range
if min(paces_min_per_km) < 4 or max(paces_min_per_km) > 7:
    log.warning("Pace values outside expected range!")

# Calculate duration
durations_min = distances_run_km * paces_min_per_km

# Heart rate - assuming relatively fit runner
heart_rates = np.random.normal(loc=160, scale=10, size=num_runs).clip(120, 180)

# Elevation - more for longer runs
elevation_gain = (distances_run_km ** 1.3) * np.random.uniform(1, 5, size=num_runs)
elevation_gain = elevation_gain.clip(10, 100).round(1)

# Run type assignment
log.info("Categorizing runs...")
run_types = []
type_counter = {}

for d, p in zip(distances_run_km, paces_min_per_km):
    if d > 15:
        run_type = "long run"
    elif p < 4.5:
        run_type = "tempo"
    elif np.random.rand() < 0.2:
        run_type = "race"
    elif np.random.rand() < 0.3:
        run_type = "intervals"
    else:
        run_type = "easy"
    
    run_types.append(run_type)
    
    # Keep track of distribution
    if run_type in type_counter:
        type_counter[run_type] += 1
    else:
        type_counter[run_type] = 1

# Print stats
for run_type, count in type_counter.items():
    log.debug(f"{run_type}: {count} ({count/len(run_types)*100:.1f}%)")

# FIXME: We're getting too many easy runs - adjust probabilities?

# Create the dataframe
df = pd.DataFrame({
    'date': dates,
    'distance_km': distances_run_km.round(2),
    'pace_min_per_km': paces_min_per_km.round(2),
    'duration_min': durations_min.round(1),
    'heart_rate_bpm': heart_rates.round(0),
    'elevation_gain_m': elevation_gain,
    'run_type': run_types
})

# Sort chronologically
df = df.sort_values('date').reset_index(drop=True)

# Sanity check
if len(df) != num_runs:
    log.error(f"Data length mismatch! Expected {num_runs}, got {len(df)}")

# Save the data
output_file = "../data/fake_running_data_with_elevation.csv"
log.info(f"Writing {len(df)} entries to {output_file}")
df.to_csv(output_file, index=False)

log.info("Done!")