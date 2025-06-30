import streamlit as st
import numpy as np
import plotly.graph_objs as go
from FROSTWAVE import run_simulation
from visualization import set_figure_layout


def main():
    st.title("Flow Rate Factor Sensitivity Analysis")
    st.write('Analyzing the impact of flow rate factors on final well head height.')

    # User inputs
    T_water = st.number_input('Water Temperature (°C)', value=18.3)
    total_sim_time_years = st.number_input('Total Simulation Time (years)', min_value=1, max_value=30, value=1)
    base_summer_flow = st.number_input('Base Summer Flow Rate (cubic meters/day)', min_value=0.0, max_value=100.0,
                                       value=11.36)
    summer_duration = st.number_input('Summer Duration (days)', min_value=0, max_value=365, value=120)
    base_winter_flow = st.number_input('Base Winter Flow Rate (cubic meters/day)', min_value=0.0, max_value=100.0,
                                       value=5.0)
    # Create a placeholder for status updates
    status_placeholder = st.empty()

    # Other fixed inputs
    total_sim_time = total_sim_time_years * 365 * 24 * 60  # convert from years to minutes
    well_length = 80
    n_vertical_slices = 100
    rw = 0.1
    Kh_index = 60 * 10 ** -6
    T_firn = -50
    hA_firn_water = 5

    # Placeholder for simulation results
    simulation_results = []

    # Factor range
    factors = np.arange(0.2, 1.6, 0.2)

    if st.button('Run Sensitivity Analysis'):
        for factor in factors:
            status_placeholder.info(f"Running simulation for water usage factor: {factor}")
            # Adjust flow rates based on the factor
            summer_flow = base_summer_flow * factor
            winter_flow = base_winter_flow * factor

            years = total_sim_time_years
            summer_rate = summer_flow / (24 * 60)
            winter_rate = winter_flow / (24 * 60)
            summer_duration_days = summer_duration

            # Initialize the list with the starting point
            switch_points_days = [(0, summer_rate)]

            # Loop through each year to add the switch points
            for y in range(years):
                # Add the switch point for the start of winter in each year
                start_winter_days = summer_duration_days + (365 * y)
                switch_points_days.append((start_winter_days, winter_rate))

                # If not the last year, add the start of the next summer
                if y < years - 1:
                    start_next_summer_days = (365 * (y + 1))
                    switch_points_days.append((start_next_summer_days, summer_rate))

            # Convert switch points to minutes for use in the function
            flow_pattern = [(time_in_days * 1440, rate) for time_in_days, rate in switch_points_days]
            flow_pattern.sort()

            # Run the simulation
            solution, z_vector, _ = run_simulation(flow_pattern, well_length, rw, n_vertical_slices,
                                            Kh_index, T_water, T_firn, hA_firn_water, total_sim_time)

            # Extract the final well head height
            final_height = solution.y[0, -1]
            final_radii = solution.y[2:2 + len(z_vector), -1]
            masked_final_radii = np.where(final_radii > rw, final_radii, np.nan)
            final_radius = np.nanmean(masked_final_radii)
            simulation_results.append((factor, final_height, final_radius))

        # Plot the results
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[result[0] for result in simulation_results],
            y=[result[1] for result in simulation_results],
            mode='markers+lines',
            name='Final Well Head Height'
        ))

        fig.add_trace(go.Scatter(
            x=[result[0] for result in simulation_results],
            y=[result[2] for result in simulation_results],
            mode='markers+lines',
            name='Final Radius'
        ))

        fig.update_layout(
            title='Impact of Flow Rate Factor on Final Well Head Height',
            xaxis_title='Flow Rate Factor',
            yaxis_title='Final Well Head Height (m)',
            legend_title='Legend',
            font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
            xaxis=dict(
                title='Flow Rate Factor',
                tickmode='linear',  # Use linear mode for ticks
                tick0=0.2,  # Start ticks at 0.2
                dtick=0.2,  # Space between ticks
            ),
            legend = dict(
                orientation="v",  # Vertical orientation
                x=0.75,  # X position in the fraction of the figure width
                y=0.60,  # Y position in the fraction of the figure height
                xanchor="center",  # Horizontal anchor of the legend is centered at x position
                yanchor="middle",  # Vertical anchor of the legend is centered at y position
                bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
                font=dict(
                    family="Arial, sans-serif",  # Set the font family you prefer
                    size=12  # Set the font size
                )
            )
        )

        set_figure_layout(fig, 'Impact of Flow Rate Factor on Final Well Head Height', 'Flow Rate Factor',
                          'Final Dimension (m)')

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
