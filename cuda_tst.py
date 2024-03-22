import plotly.graph_objects as go
import numpy as np

# Generate some sample data
x_values = np.arange(0, 10)
y_values = np.arange(0, 10)
z_values = np.random.rand(10, 10)

# Additional custom data (random numbers with some NaN values)
custom_data = np.random.rand(10, 10)
custom_data[np.random.choice([True, False], size=custom_data.shape, p=[0.1, 0.9])] = np.nan

# Additional custom labels (categorical)
custom_labels = np.random.choice(['A', 'B', 'C', 'D'], (10, 10))

# Create the heatmap trace
heatmap_trace = go.Heatmap(
    x=x_values,
    y=y_values,
    z=z_values,
    customdata=np.stack((custom_data, custom_labels), axis=-1),  # Combine custom data and labels
    hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Z:</b> %{z}<br><b>Custom Data:</b> %{customdata[0]}<br><b>Label:</b> %{customdata[1]}',  # Define hover template
)

# Create the figure and add the trace
fig = go.Figure(data=[heatmap_trace])

# Update layout if necessary
fig.update_layout(
    title='Fake Heatmap with Custom Data and Labels (NaN values)',
    xaxis_title='X Axis',
    yaxis_title='Y Axis'
)

# Display the figure
fig.show()
