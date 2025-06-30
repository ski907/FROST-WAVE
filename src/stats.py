import pandas as pd
import numpy as np


def calculate_summary_stats(solution, z_vector, rw, time_intervals):
    # Assuming the simulation times are in minutes, convert to years for indexing
    years = np.array(time_intervals) / (60 * 24 * 365)

    # Extract radii from the solution
    radii = solution.y[2:2 + len(z_vector), :]
    well_heads = solution.y[0, :]

    # Mask the radii that are less than or equal to rw with np.nan
    masked_radii = np.where(radii > rw, radii, np.nan)

    # Calculate average radii while ignoring NaN values
    average_radii = np.nanmean(masked_radii, axis=0)
    std_radii = np.nanstd(masked_radii, axis=0)
    max_radii = np.nanmax(masked_radii, axis=0)

    summary_df = pd.DataFrame({
        'Year': years,
        'Average Radius (m)': average_radii,
        'Standard Deviation of Radius (m)': std_radii,
        'Max Radius (m)': max_radii,
        'Well Head Height (m)': well_heads
    })

    # Remove rows with NaN values that may occur if all radii are less than or equal to rw
    summary_df.dropna(inplace=True)

    return summary_df