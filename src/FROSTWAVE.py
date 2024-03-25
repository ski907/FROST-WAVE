import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import jit
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#@jit(nopython=True)
def derivatives(t, y, z_vector, rw, dz, rho_reference, Kh_index, Kv, h_firn_water, delta_theta, psi, T_firn, flow_pattern):

    # Determine the current flow rate
    Q_in = get_current_flow_rate(t, flow_pattern)

    # Unpack the state vector y into variables
    hw = y[0]
    Z = y[1]
    R = y[2:2 + len(z_vector)]
    T_sat_front = y[2 + len(z_vector):2 + 2 * len(z_vector)]
    rho_vector = y[2 + 2 * len(z_vector):]

    # Initialize the derivatives for density to zero
    drho_dt = np.zeros_like(T_sat_front)

    dT_sat_front_dt, frozen_mass_max = calculate_dT_sat_front_dt(T_firn, T_sat_front, R, rw, h_firn_water)
    # Ensure that T_sat_front does not go below 0 by zeroing its derivative if it's at 0
    dT_sat_front_dt[T_sat_front <= 0] = 0

    # Only proceed with the following calculations if T_sat_front is 0 degree C
    freezing_temp = 0  # degree C
    tolerance = 1e-3  # a small tolerance to check for equality to handle numerical approximations

    if np.any(np.abs(T_sat_front - freezing_temp) < tolerance):
        Ck = 1  # freezing calibration constant
        dmass_freezing_dt = Ck * frozen_mass_max
        drho_dt = dmass_freezing_dt

        # Update the densities only where the temperature is around 0 degree C
        drho_dt[np.abs(T_sat_front - freezing_temp) >= tolerance] = 0
    max_rho = 1000
    drho_dt[rho_vector >= max_rho] = 0

    rho_vector+=drho_dt
    Kh = calculate_Kh(rho_vector, rho_reference, Kh_index)


    #solve for dR/dt, dZ/dt and dhw/dt
    # first solve for Qh
    Q_h, R = calculate_Q_h(z_vector, R, hw, rw, Kh, dz, psi)
    #then Qv
    #Q_v = calculate_Q_v(z_vector, R, rw, Kv)
    #some stability issues, so zeroing this for now
    Q_v = np.zeros_like(Q_h)

    #now calulate the change in radiuses
    dR_dt = calculate_dR_dt(Q_h, Q_v, R, dz, delta_theta)

    #experiment, seems to work but is slow
    #basically explicit euler method
    R_new = R + dR_dt
    #R_new = R

    #Change in the well height
    dhw_dt = calculate_dhw_dt(Q_in, Q_v, delta_theta, R, R_new, dz, rw)

    #and the change in Z
    dZ_dt = calculate_dZ_dt(Kv, delta_theta)

    # Pack the derivatives into a single array and return
    result = np.empty(2 + 3*len(z_vector))

    # Fill the result array
    result[0] = dhw_dt
    result[1] = dZ_dt
    result[2:2+len(z_vector)] = dR_dt
    result[2+len(z_vector):2+2*len(z_vector)] = dT_sat_front_dt
    #result[2+2*len(z_vector):] = drho_vector_dt
    result[2 + 2 * len(z_vector):] = drho_dt

    return result


@jit(nopython=True)
def calculate_dT_sat_front_dt(T_firn, T_sat_front, R, rw, hA_firn_water):
    # Parameters related to heat transfer
    #hA_firn_water = 100  # Heat transfer coefficient times specific area, W/m²K-m (this is a placeholder, adjust as needed)
    c = 4184  # Specific heat capacity of water, J/kgK
    rho = 1000  # Density of water, kg/m³
    latent_heat_fusion = 334 * 1000 #J/kg


    # Compute the derivative of T_sat_front, dT_sat_front_dt
    dT_sat_front_dt = np.zeros_like(T_sat_front)  # initialize with zeros

    # Initialize frozen_mass_max as an array of zeros
    frozen_mass_max = np.zeros_like(T_sat_front)

    # Check if R > rw for each element, and if true, calculate the corresponding frozen_mass_max
    for i in range(len(R)):
        if R[i] > rw:
            #this is Newtons Law of Cooling where Q = h*A*(Ti-T)
            #h is the heat transfer coefficient (usually W/m^2K)
            #A is the specific Area, that is a function of the porous structure
            dT_sat_front_dt[i] = hA_firn_water * (T_firn - T_sat_front[i]) / (c * rho)  # Only update if R > rw
            frozen_mass_max[i] = - (T_firn * c * rho) / latent_heat_fusion  # See Illangasekare et al 1990
        # If R <= rw, both dT_sat_front_dt[i] and frozen_mass_max[i] remain zero

    return dT_sat_front_dt, frozen_mass_max

@jit(nopython=True)
def calculate_Q_h(z_vector, R, hw, rw, Kh, dz, psi):
    R_corrected = R
    Q_h = np.zeros_like(z_vector)
    z_below_water = z_vector < hw
    if np.any(z_below_water):
        # locate the last True value in the z_below_water array and set it to False
        z_below_water[np.argwhere(z_below_water)[-1][0]] = False

    for i, z in enumerate(z_vector[z_below_water]):
        eps_r = R[i] / rw
        if eps_r < 1.01:
            eps_r = 1.01
            R_corrected[i] = rw * eps_r #not sure if this is needed
        Q_h[i] = 2 * np.pi * dz * Kh[i] * (hw - z - psi[i]) / (np.log(eps_r))

    return Q_h, R_corrected

@jit(nopython=True)
def calculate_Q_v(z_vector, R, rw, Kv):
    Q_v = np.zeros_like(R)

    for i, z in enumerate(z_vector):
        if R[i] > rw:
            Q_v[i] = -np.pi*(R[i]**2-rw**2) * Kv[i]
        #net_Q_v = np.append(-np.diff(Q_v_down), 0) #net flow is zero at the top?

    return Q_v

@jit(nopython=True)
def calculate_dR_dt(Q_h, Q_v, R, dz, delta_theta):
    dR_dt = np.zeros_like(R)
    Q_v_in = np.roll(Q_v, shift=1)
    Q_v_in[-1] = 0

    for i in range(len(R)):
        dR_dt[i] = (Q_h[i] - Q_v[i] + Q_v_in[i]) / (2 * np.pi * R[i] * dz * delta_theta[i])

        # Ensure dR_dt is never negative
        dR_dt[i] = max(0, dR_dt[i])

    return dR_dt

@jit(nopython=True)
def calculate_dZ_dt(Kv, delta_theta):

    dZ_dt = Kv[0]/delta_theta[-1]

    return dZ_dt

@jit(nopython=True)
def calculate_dhw_dt(Q_in, Q_v, delta_theta, R, R_new, dz, rw):
    change_in_saturated_zone = 0
    for i in range(len(R)):
        change_in_saturated_zone += (R_new[i]**2 - R[i]**2) * np.pi * delta_theta[i] * dz
    dhw_dt = (Q_in - Q_v[0] - change_in_saturated_zone) / (np.pi * rw**2)

    return dhw_dt

# @jit(nopython=True)
# def calculate_Kh(z_vector, rho_top, rho_bottom, Kh_index):
#     well_length = z_vector[-1]  # assuming z_vector starts from 0
#     rho_snow = rho_top + (z_vector / well_length) * (rho_bottom - rho_top)
#     Kh = Kh_index * np.exp(-7.8 * rho_snow/1000)/ np.exp(-7.8 * rho_bottom/1000)
#     print(Kh)
#     return Kh


@jit(nopython=True)
def calculate_Kh(rho_vector, rho_reference, Kh_index):
    """
    Calculate Kh values based on a vector of snow densities.

    Parameters:
    - rho_vector: Array of snow density values
    - rho_reference: A specific snow density to which Kh_index is mapped
    - Kh_index: Index permeability value mapped to rho_reference

    Returns:
    - Kh: Array of hydraulic conductivity values
    """
    # Ensure that rho_reference is not mapped to zero in the exponential function
    assert rho_reference > 0, "rho_reference should be > 0 to avoid division by zero"

    # Calculate Kh based on the provided snow densities
    #based on Shumizu 1970
    Kh = Kh_index * np.exp(-7.8 * rho_vector / 1000) / np.exp(-7.8 * rho_reference / 1000)
    Kh[rho_vector >= 1000] = 0
    #print(Kh)
    return Kh


# Data from the table
depths = [0, 20, 100]
densities = [400, 550, 800]

# Create interpolation function
interpolation_function = interp1d(depths, densities, kind='linear', fill_value="extrapolate")

def get_density_at_depth(depth):
    """Returns the density at the specified depth using linear interpolation."""
    return interpolation_function(depth)

def get_current_flow_rate(t, flow_pattern):
    # Assuming flow_pattern is a list of tuples (time_in_minutes, flow_rate)
    # and is sorted by time_in_minutes.
    # Default to the first flow rate if t is before the first change point
    current_flow_rate = flow_pattern[0][1]

    for change_point, flow_rate in flow_pattern:
        if t >= change_point:
            current_flow_rate = flow_rate
        else:
            break  # Since the list is sorted, we can exit early

    return current_flow_rate

def plot_results_plotly(solution, z_vector, Kh):

    # Convert time from minutes to years for the first subplot
    t_sol_years = solution.t / (60 * 24 * 365.25)
    # Extract the results from the solution
    t_sol = solution.t
    hw_sol = solution.y[0, :]
    Z_sol = solution.y[1, :]
    R_sol = solution.y[2:2 + len(z_vector), :]
    T_sat_front_sol = solution.y[2 + len(z_vector):2 + 2 * len(z_vector), :]
    rho_sol = solution.y[2 + 2 * len(z_vector):, :]

    # Create a figure with three subplots
    fig = make_subplots(rows=4, cols=1, subplot_titles=('Outfall Head (m) and Well Elevation (m)', 'Radius of Outfall', 'Snow Density With Depth'))

    # First subplot: hw_sol and Z_sol against time
    fig.add_trace(go.Scatter(x=t_sol_years, y=hw_sol, mode='lines', name='hw_sol'), row=1, col=1)

    # Second subplot: each row of R_sol against z_vector
    num_traces = 6
    indices_to_plot = np.linspace(0, R_sol.shape[1] - 1, num_traces, dtype=int)
    for i in indices_to_plot:
        fig.add_trace(go.Scatter(x=R_sol[:, i], y=z_vector, mode='lines', name=f'Time {t_sol_years[i]:.2f} years'), row=2, col=1)

  #  # Third subplot: T_sat_front_sol against z_vector
   # for i in range(T_sat_front_sol.shape[1]):
    #    fig.add_trace(go.Scatter(x=T_sat_front_sol[:, i], y=z_vector, mode='lines', name=f'Time {t_sol[i]}'), row=3, col=1)

    fig.add_trace(go.Scatter(x=Kh, y=z_vector, mode='lines', name='Kh'), row=3, col=1)

    # Update xaxis and yaxis labels
    fig.update_xaxes(title_text='Time (years)', row=1, col=1)
    fig.update_yaxes(title_text='Elevation (m)', row=1, col=1)
    fig.update_xaxes(title_text='Radius (m)', row=2, col=1)
    fig.update_yaxes(title_text='Elevation (m)', row=2, col=1)
    fig.update_xaxes(title_text='Kh (m/min)', row=3, col=1)
    fig.update_yaxes(title_text='Elevation (m)', row=3, col=1)

    # Update xaxis properties to ensure vertical grid lines are shown
    fig.update_xaxes(showgrid=True, gridcolor='LightGrey', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='LightGrey', row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='LightGrey', row=3, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='LightGrey', row=4, col=1)

    # Update layout for a cleaner look
    fig.update_layout(height=2000, width=1000, title_text='Simulation Results')
    # Show the plot if running in a notebook; in Streamlit, return the figure object
    # fig.show()
    return fig

def plot_separate_results_plotly(solution, z_vector, Kh, total_sim_time_years):
    # Convert time from minutes to years for the subplot
    t_sol_years = solution.t / (60 * 24 * 365.25)

    # Extract the results from the solution
    hw_sol = solution.y[0, :]
    Z_sol = solution.y[1, :]
    R_sol = solution.y[2:2 + len(z_vector), :]
    T_sat_front_sol = solution.y[2 + len(z_vector):2 + 2 * len(z_vector), :]
    rho_sol = solution.y[2 + 2 * len(z_vector):, :]

    figures = []

    # First subplot: hw_sol against time
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t_sol_years, y=hw_sol, mode='lines', name='hw_sol'))
    fig1.update_layout(title='Outfall Head (m)', xaxis_title='Time (years)', yaxis_title='Elevation (m)', height=600, width=800)
    fig1.update_xaxes(showgrid=True, gridcolor='LightGrey')
    figures.append(fig1)

    # Second subplot: each row of R_sol against z_vector
    fig2 = go.Figure()
    num_traces = total_sim_time_years +1
    indices_to_plot = np.linspace(0, R_sol.shape[1] - 1, num_traces, dtype=int)
    for i in indices_to_plot:
        fig2.add_trace(go.Scatter(x=R_sol[:, i], y=z_vector, mode='lines', name=f'Time {t_sol_years[i]:.2f} years'))
    fig2.update_layout(title='Radius of Outfall', xaxis_title='Radius (m)', yaxis_title='Elevation (m)', height=600, width=800)
    fig2.update_xaxes(showgrid=True, gridcolor='LightGrey')
    figures.append(fig2)

    # Identify the last index for each year and create the figures
    years = np.arange(1, total_sim_time_years + 1)
    last_indices_each_year = [np.searchsorted(t_sol_years, year, side='right') - 1 for year in years]

    for year, index in zip(years, last_indices_each_year):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=R_sol[:, index], y=z_vector, mode='lines', name=f'End of Year {year}'))

        # Use the set_figure_layout function to standardize the layout across all figures
        set_figure_layout(fig, f'Radius of Outfall at the End of Year {year}', 'Radius (m)', 'Elevation (m)')
        figures.append(fig)

    # Third subplot: Kh against z_vector
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Kh, y=z_vector, mode='lines', name='Kh'))
    fig3.update_layout(title='Hydraulic Conductivity With Depth', xaxis_title='Kh (m/min)', yaxis_title='Elevation (m)', height=600, width=800)
    fig3.update_xaxes(showgrid=True, gridcolor='LightGrey')
    figures.append(fig3)

    # Apply the layout settings to all the figures using the same function
    set_figure_layout(fig1, 'Outfall Head (m)', 'Time (years)', 'Elevation (m)')
    set_figure_layout(fig2, 'Radius of Outfall', 'Radius (m)', 'Elevation (m)')
    set_figure_layout(fig3, 'Hydraulic Conductivity With Depth', 'Kh (m/min)', 'Elevation (m)')


    # Return the list of figures
    return figures

def set_figure_layout(figure, title, xaxis_title, yaxis_title):
    title_font_size = 24
    axis_title_font_size = 20
    axis_tick_font_size = 14

    figure.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=title_font_size,
                color='black',
                family='Arial Bold, sans-serif',
            ),
        },
        xaxis={
            'title': xaxis_title,
            'title_font': {
                'size': axis_title_font_size,
                'color': 'black',
                'family': 'Arial Bold, sans-serif',
            },
            'tickfont': dict(
                size=axis_tick_font_size,
                color='black',
                family='Arial Bold, sans-serif',
            ),
            'showgrid': True,
            'gridcolor': 'LightGrey'
        },
        yaxis={
            'title': yaxis_title,
            'title_font': {
                'size': axis_title_font_size,
                'color': 'black',
                'family': 'Arial Bold, sans-serif',
            },
            'tickfont': dict(
                size=axis_tick_font_size,
                color='black',
                family='Arial Bold, sans-serif',
            ),
            'showgrid': True,
            'gridcolor': 'LightGrey'
        },
        height=600,
        width=800
    )



def run_simulation(flow_pattern, well_length=100, rw=0.1, n_vertical_slices=100,
                   Kh_index=60 * 10**-6, T_water=5, T_firn=-40,
                   hA_firn_water=100, total_sim_time=60*24*365/20):


    # adopting the convention of moreno z = 0 at bottom of well
    z_vector = np.linspace(0, well_length, n_vertical_slices)
    dz = well_length / (n_vertical_slices - 1)

    # Specify the index hydraulic conductivity
    Kh_index *= 60 #convert to m/min

    # Calculate a linear gradient of snow densities along the well
    #initial_rho_vector = np.linspace(rho_bottom, rho_top, len(z_vector))
    initial_rho_vector = get_density_at_depth(z_vector[::-1]) #reversed order of z_vector

    # Specify a reference density for which Kh_index is defined
    #rho_reference = rho_bottom
    rho_reference = initial_rho_vector[-1] #kg/m^3 tied to the density at the surface

    # Calculate Kh for each z
    Kh = calculate_Kh(initial_rho_vector, rho_reference, Kh_index)
    #print(Kh)

    #constant Kv
    Kv = np.full_like(z_vector, 10**-7)
    psi = np.full_like(z_vector, 0)

    delta_theta_vector = (1000 - initial_rho_vector)/1000
    delta_theta = 0.5

    t_start = 0
    t_end = total_sim_time

    #time_intervals = np.linspace(0, total_sim_time, 10)
    time_intervals = np.arange(0, total_sim_time, (30*24*60))

    # Initial conditions
    initial_hw = 0
    initial_Z = well_length
    initial_R = np.full_like(z_vector, rw)
    initial_T_sat_front = np.full_like(z_vector, T_water)

    # Pack initial conditions into a single array
    y0 = np.concatenate(([initial_hw, initial_Z], initial_R, initial_T_sat_front, initial_rho_vector))


    # Call solve_ivp
    solution = solve_ivp(
        fun=derivatives,
        t_span=(t_start, t_end),
        y0=y0,
        args=(z_vector, rw, dz, rho_reference, Kh_index, Kv, hA_firn_water, delta_theta_vector, psi, T_firn, flow_pattern),
        method='RK23',
        t_eval=time_intervals,
        #atol=1e-3,
        #rtol=1e-3,
    )

    if not solution.success:
        print("Error: Solver did not successfully find a solution.")
        print("Solver message:", solution.message)
    else:
        print("Sucessfull solve!")

    return solution, z_vector, Kh
    #return plot_results_plotly(solution,z_vector)

def main():
    well_length = 100  # m
    n_vertical_slices = 100
    # adopting the convention of moreno z = 0 at bottom of well
    z_vector = np.linspace(0, well_length, n_vertical_slices)
    dz = well_length / (n_vertical_slices - 1)

    # Specify the top and bottom snow densities
    rho_top = 400  # kg/m^3
    rho_bottom = 700  # kg/m^3

    # Specify the index hydraulic conductivity
    Kh_index = 60 * 10 ** -6  # m/s
    Kh_index *= 60 #convert to m/min

    # Calculate a linear gradient of snow densities along the well
    initial_rho_vector = np.linspace(rho_bottom, rho_top, len(z_vector))

    # Specify a reference density for which Kh_index is defined
    rho_reference = rho_bottom  # or another value as per your requirements

    # Calculate Kh for each z
    Kh = calculate_Kh(initial_rho_vector, rho_reference, Kh_index)

    #constant Kv
    Kv = np.full_like(z_vector, 10**-7)

    psi = np.full_like(z_vector, 0)
    rw = 0.1

    delta_theta = 0.3
    #inflow rate
    Q_in = 8.18/(24*60*60)*60 * 100  #m^3/minute
    T_water = 5 #degrees C
    T_firn = -40 #degrees C
    hA_firn_water = 100 #J/min-m^2-K-m heat transfer coefficient times specific area

    total_sim_time = 60*24*365  /20   #minutes

    t_start = 0
    t_end = total_sim_time

    time_intervals = np.linspace(0, total_sim_time, 10)

    # Initial conditions
    initial_hw = 0
    initial_Z = well_length
    initial_R = np.full_like(z_vector, rw)
    initial_T_sat_front = np.full_like(z_vector, T_water)

    # Pack initial conditions into a single array
    y0 = np.concatenate(([initial_hw, initial_Z], initial_R, initial_T_sat_front, initial_rho_vector))

    # Example structure: [(duration_in_minutes, flow_rate), ...]
    time_flow_pairs = [
        (120 * 24 * 60, 8.18/(24*60)),  # 120 days of summer flow rate
        (245 * 24 * 60, 8.18/(24*60)),  # 245 days of winter flow rate
        # Add more periods as needed
    ]

    # Call solve_ivp
    solution = solve_ivp(
        fun=derivatives,
        t_span=(t_start, t_end),
        y0=y0,
        args=(z_vector, rw, dz, rho_reference, Kh_index, Kv, hA_firn_water, delta_theta, psi, T_firn, time_flow_pairs),
        method='RK23',
        t_eval=time_intervals,
        #atol=1e-3,
        #rtol=1e-3,
    )

    if not solution.success:
        print("Error: Solver did not successfully find a solution.")
        print("Solver message:", solution.message)
    else:
        print("Sucessfull solve!")

    plot_results(solution,z_vector)


if __name__ == "__main__":
    main()