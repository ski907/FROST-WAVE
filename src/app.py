##FROST-WAVE
#Firn Radial Outfall Simulation Tool for Wastewater Analysis and Volumetric Estimation


import streamlit as st
import numpy as np
from FROSTWAVE import run_simulation
from visualization import plot_separate_results_plotly
from stats import calculate_summary_stats

def main():
    st.title("FROST-WAVE")
    st.write('Firn Radial Outfall Simulation Tool for Wastewater Analysis and Volumetric Estimation')
    total_sim_time_years = st.number_input('Total Simulation Time (years)', min_value=1, max_value=30,
                                     value=1)
    total_sim_time = total_sim_time_years * 365 * 60 * 24  # convert from years to minutes
    well_length = st.number_input('Well Length (m)', min_value=1, max_value=100, value=40)
    n_vertical_slices = st.number_input('Number of Vertical Slices', min_value=1, max_value=500, value=100)
    rw = st.number_input('Well Radius (m)', min_value=0.0, max_value=100.0, value=0.1)
    summer_flow = st.number_input('Summer Flow Rate (cubic meters/day)', min_value=0.0, max_value=100.0, value=7.6)
    summer_duration = st.number_input('Summer Duration (days)', min_value=0, max_value=365, value=92)
    winter_flow = st.number_input('Winter Flow Rate (cubic meters/day)', min_value=0.0, max_value=100.0, value=2.3)


    with st.sidebar:
        # Expose variables



        # Specify the top and bottom snow densities
        #rho_top = st.number_input('Snow Density at top (kg/m^3)', min_value=1, max_value=1000, value=500)  # kg/m^3
        #rho_bottom = st.number_input('Snow Density at bottom (kg/m^3)', min_value=1, max_value=1000, value=640)  # kg/m^3
        Kh_index = st.number_input('Hydraulic Conductivity at surface  (x 10^-6 m/s)',min_value=1, max_value=1000000, value = 60)  # x10^-6m/s
        Kh_index *= 10 ** -6

        #inflow
        #Q_in = st.number_input('Inflow Rate (m^3/day)',min_value=float(1), max_value=float(100000), value = float(8.18) )  #m^3/day
        #Q_in /= (24*60) #convert to m^3/min
        T_water = st.number_input('Water Temperature (C)', min_value=-40.0, max_value=100.0, value=18.3) #degrees C
        T_firn = st.number_input('Firn Temperature (C)', min_value=-150, max_value=100, value=-50) #degrees C
        hA_firn_water = st.number_input('Heat Transfer Coefficient Times Specific Area (J/(min-K))', min_value=1, max_value=1000, value=5) #J/(min-K) heat transfer coefficient times specific area


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


        # Sort the pattern by time since we're appending in order but handling summer and winter separately
        flow_pattern.sort()

    # Button to run the simulation
    if st.button('Run Simulation'):
        solution, z_vector, Kh = run_simulation(flow_pattern, well_length,rw=rw, n_vertical_slices=n_vertical_slices,
                             Kh_index=Kh_index, T_water=T_water, T_firn=T_firn, hA_firn_water=hA_firn_water,
                             total_sim_time=total_sim_time)

        # Calculate summary statistics
        time_intervals = solution.t
        summary_stats_df = calculate_summary_stats(solution, z_vector, rw, time_intervals)


        figures = plot_separate_results_plotly(solution, z_vector, Kh, total_sim_time_years)
        for fig in figures:
            st.plotly_chart(fig)

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(summary_stats_df, width=700, height=600)




if __name__ == "__main__":
    main()
