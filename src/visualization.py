import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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

    # Define the theta grid for all plots
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z_vector)

    #Initialize the figure with layout options
    fig3d = go.Figure()

    # Add frames for each timestep (assuming 10 increments)
    num_timesteps = R_sol.shape[1]
    timesteps_to_show = np.linspace(0, num_timesteps - 1, 25, dtype=int)

    frames = []  # Initialize an empty list for frames
    for i, timestep in enumerate(timesteps_to_show):
        r_grid = np.repeat(R_sol[:, timestep][:, np.newaxis], len(theta), axis=1)
        x = r_grid * np.cos(theta_grid)
        y = r_grid * np.sin(theta_grid)

        frame = go.Frame(data=[go.Surface(z=z_grid, x=x, y=y)],
                         name=str(t_sol_years[timestep]))
        fig3d.add_trace(
            go.Surface(z=z_grid, x=x, y=y, showscale=False, visible=(i == 0)))

        frames.append(frame)

    fig3d.frames = frames
    # Set the first timestep visible
    fig3d.data[0].visible = True
    z_base = max(z_vector) - 15.2
    height = 2  # Height of the box

    # Top and bottom faces
    z_top = z_base
    z_bottom = z_base - height

    # Define corner points for each face
    # Note: Plotly draws surfaces between the provided corner points.
    x_corners = [-1, 1, 1, -1, -1]  # Loop back to the start to complete the rectangle
    y_corners_side = [-20, -20, 0, 0, -20]  # For the side faces
    y_corners_top = [0, 0, -20, -20, 0]  # For the top and bottom faces
    z_corners_top = [z_top, z_top, z_top, z_top, z_top]  # For the top face
    z_corners_bottom = [z_bottom, z_bottom, z_bottom, z_bottom, z_bottom]  # For the bottom face

    # Add the top face
    fig3d.add_trace(go.Surface(x=x_corners, y=y_corners_top, z=z_corners_top, showscale=False, opacity=0.5,
                               colorscale=[[0, 'blue'], [1, 'blue']]))

    # Add the bottom face
    fig3d.add_trace(go.Surface(x=x_corners, y=y_corners_top, z=z_corners_bottom, showscale=False, opacity=0.5,
                               colorscale=[[0, 'blue'], [1, 'blue']]))

    # Add the four side faces
    for i in range(4):
        fig3d.add_trace(go.Surface(x=[x_corners[i], x_corners[i + 1], x_corners[i + 1], x_corners[i]],
                                   y=[y_corners_side[i], y_corners_side[i + 1], y_corners_side[i + 1],
                                      y_corners_side[i]],
                                   z=[z_top, z_top, z_bottom, z_bottom],
                                   showscale=False,
                                   opacity=0.5,
                                   colorscale=[[0, 'blue'], [1, 'blue']]))

    # Create and add slider
    steps = []
    for i, timestep in enumerate(timesteps_to_show):
        step = dict(
            method="animate",
            args=[[str(t_sol_years[timestep])],
                  {"frame": {"duration": 300, "redraw": True},
                   "mode": "immediate",
                   "transition": {"duration": 300}}],
            label=f"{t_sol_years[timestep]:.2f}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Decimal Year: "},
        pad={"t": 50},
        steps=steps
    )]

    fig3d.update_layout(
        title='Well Radius Variation with Depth Over Time', height=600, width=800,
        sliders=sliders,
        scene=dict(
            zaxis=dict(title='Depth'),
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        )
    )

    # Determine the maximum radius to set x and y axis limits
    max_radius = np.max(R_sol)

    # Set z axis limits based on z_vector
    min_z, max_z = np.min(z_vector), np.max(z_vector)

    # Update the layout of your figure to set the axis limits
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(range=[-max_radius, max_radius]),
            yaxis=dict(range=[-max_radius, max_radius]),
            zaxis=dict(range=[min_z, max_z]),
            aspectmode='cube'
            # Optional: Forces equal aspect ratio for all axes, making the plot more visually accurate
        )
    )

    figures.append(fig3d)


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

    max_y = max(z_vector)  # Find the maximum y-value to determine the top of the figure

    fig2.add_hline(
        y=max_y - 15.2,
        line_dash="dot",
        annotation_text="Bottom of Tunnel",
        annotation_position="bottom right",
    )

    figures.append(fig2)

    # Identify the last index for each year and create the figures
    years = np.arange(1, total_sim_time_years + 1)
    last_indices_each_year = [np.searchsorted(t_sol_years, year, side='right') - 1 for year in years]

    for year, index in zip(years, last_indices_each_year):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=R_sol[:, index], y=z_vector, mode='lines', name=f'End of Year {year}'))

        # Use the set_figure_layout function to standardize the layout across all figures
        set_figure_layout(fig, f'Radius of Outfall at the End of Year {year}', 'Radius (m)', 'Elevation (m)')
        # Add a horizontal line at 15.2m down from the top
        max_y = max(z_vector)  # Find the maximum y-value to determine the top of the figure

        fig.add_hline(
            y=max_y - 15.2,
            line_dash="dot",
            annotation_text="Bottom of Tunnel",
            annotation_position="bottom right",
        )
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