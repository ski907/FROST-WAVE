import streamlit as st
import numpy as np
import plotly.graph_objs as go
from FROSTWAVE import run_simulation
from stats import calculate_summary_stats
from visualization import set_figure_layout


def main():
    st.title("FROST-WAVE Sensitivity Analysis")
    st.write('Analyzing the impact of water temperature on final average radius and well head height.')

    # Get a string of temperatures from the user
    temperature_input = st.text_input('Enter Temperatures (°C) separated by commas', '5, 15, 25')

    # Convert the string input to a list of integers
    try:
        temperatures = [int(temp.strip()) for temp in temperature_input.split(',')]
    except ValueError:
        st.error("Please enter valid integers separated by commas.")
        st.stop()

    # Define other fixed inputs
    total_sim_time_years = st.number_input('Total Simulation Time (years)', min_value=1, max_value=30, value=1)
    total_sim_time = total_sim_time_years * 365 * 24 * 60  # convert from years to minutes
    well_length = 80
    n_vertical_slices = 100
    rw = 0.1
    Kh_index = 60 * 10 ** -6
    T_firn = -50
    hA_firn_water = 5
    summer_flow = 11.36   #  cubic meters/day
    winter_flow = 5   # cubic meters/day
    summer_duration = 120

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

    # Placeholder for simulation results
    simulation_results = []
    solutions = []
    z_vectors = []

    # Create a placeholder for status updates
    status_placeholder = st.empty()

    if st.button('Run Sensitivity Analysis'):
        # Loop over each temperature and run simulations
        for T_water in temperatures:
            status_placeholder.info(f"Running simulation for water temperature: {T_water} °C")

            solution, z_vector, Kh = run_simulation(flow_pattern, well_length, rw=rw, n_vertical_slices=n_vertical_slices,
                                            Kh_index=Kh_index, T_water=T_water, T_firn=T_firn,
                                            hA_firn_water=hA_firn_water, total_sim_time=total_sim_time)

            solutions.append(solution)
            z_vectors.append(z_vector)
            # Calculate summary statistics for the last time point (end of the simulation)
            final_radii = solution.y[2:2 + len(z_vector), -1]
            masked_final_radii = np.where(final_radii > rw, final_radii, np.nan)
            final_radius = np.nanmean(masked_final_radii)
            final_height = solution.y[0, -1]

            # Append the results
            simulation_results.append((T_water, final_radius, final_height))
            status_placeholder.success(f"Completed simulation for water temperature: {T_water} °C")

        # After all simulations, display the final status
        status_placeholder.success("Sensitivity analysis completed.")

        # Plot the results
        fig = go.Figure()

        # Add traces for final average radius and outfall height for each temperature
        fig.add_trace(go.Scatter(
            x=[temp for temp, _, _ in simulation_results],
            y=[radius for _, radius, _ in simulation_results],
            mode='lines+markers',
            name='Final Average Radius (m)'
        ))

        fig.add_trace(go.Scatter(
            x=[temp for temp, _, _ in simulation_results],
            y=[height for _, _, height in simulation_results],
            mode='lines+markers',
            name='Final Well Head Height (m)'
        ))

        fig.update_layout(
            title='Sensitivity Analysis of Water Temperature',
            xaxis_title='Water Temperature (°C)',
            yaxis_title=f'Final Dimension after {total_sim_time_years} years (m)',
            legend_title='Measurements',
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            ),
            legend=dict(
                orientation="v",  # Vertical orientation
                x=0.75,  # X position in the fraction of the figure width
                y=0.75,  # Y position in the fraction of the figure height
                xanchor="center",  # Horizontal anchor of the legend is centered at x position
                yanchor="middle",  # Vertical anchor of the legend is centered at y position
                bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
                font=dict(
                    family="Arial, sans-serif",  # Set the font family you prefer
                    size=12  # Set the font size
                )
            )
        )
        set_figure_layout(fig, 'Sensitivity Analysis of Water Temperature', 'Water Temperature (°C)', f'Final Dimension after {total_sim_time_years} years (m)')

        # Add a horizontal line at 15.2m down from the top
        max_y = max(z_vectors[-1])  # Find the maximum y-value to determine the top of the figure

        fig.add_hline(
            y=max_y - 15.2,
            line_dash="dot",
            annotation_text="Bottom of Tunnel",
            annotation_position="bottom right",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Initialize a figure for final geometries
        final_geometry_fig = go.Figure()

        # Loop through each result to add to the final geometries plot
        # Define a list of dash types available in Plotly
        dash_types = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        for index, (solution, T_water) in enumerate(zip(solutions, temperatures)):
            R_sol = solution.y[2:2 + len(z_vectors[-1]), :]
            dash_type = dash_types[index % len(dash_types)]  # Cycle through dash_types

            # Add a trace for this temperature's final geometry
            final_geometry_fig.add_trace(go.Scatter(
                x=R_sol[:, -1],
                y=z_vectors[-1],
                mode='lines',
                name=f'Temp {T_water}°C',
                line = dict(dash=dash_type)  # Set the dash type for the line
            ))


        # Add a horizontal line at 15.2m down from the top
        final_geometry_fig.add_hline(
            y=max_y - 15.2,
            line_dash="dot",
            annotation_text="Bottom of Tunnel",
            annotation_position="bottom right",
        )

        # Set layout for the final geometries plot
        final_geometry_fig.update_layout(
            title='Final Geometry of Outfall for Different Temperatures',
            xaxis_title='Radius (m)',
            yaxis_title='Elevation (m)',
            legend_title='Temperature',
            legend=dict(
                orientation="v",  # Vertical orientation
                x=0.8,  # X position in the fraction of the figure width
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

        set_figure_layout(final_geometry_fig, 'Final Geometry of Outfall for Different Temperatures', 'Radius (m)',
                          'Elevation (m)')
        # Display the final geometries plot
        st.plotly_chart(final_geometry_fig, use_container_width=True)

if __name__ == "__main__":
    main()
