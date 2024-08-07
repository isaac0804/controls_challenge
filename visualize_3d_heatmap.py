import pandas as pd
import numpy as np
import plotly.graph_objects as go

results_df = pd.read_csv('pid_grid_search_results.csv')
cost_type = 'avg_total_cost'
# cost_type = 'avg_lataccel_cost'
# cost_type = 'avg_jerk_cost'

plot_type = "log"
# plot_type = "linear"

x = np.log10(results_df['kp']) if plot_type=="log" else results_df['kp']
y = np.log10(results_df['ki']) if plot_type=="log" else results_df['ki']
z = -np.log10(np.abs(results_df['kd'])) if plot_type=="log" else results_df['kd']

def norm(x):
    x = np.log10(x)
    return (x-np.min(x)) / (np.max(x)-np.min(x))


fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        # color=results_df[cost_type],
        color=norm(results_df[cost_type]),
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title=cost_type)
    ),
    hovertemplate='<b>kp</b>: %{x}<br>' +
                    '<b>ki</b>: %{y}<br>' +
                    '<b>kd</b>: %{z}<br>' +
                    '<b>Avg Lataccel Cost</b>: %{text}<br>' +
                    '<b>Avg Jerk Cost</b>: %{customdata[0]:.4f}<br>' +
                    '<b>Avg Total Cost</b>: %{marker.color:.4f}<extra></extra>',
    text=[f'{cost:.4f}' for cost in results_df['avg_lataccel_cost']],
    customdata=results_df[['avg_jerk_cost']].values
)])

fig.update_layout(
    scene=dict(
        xaxis_title='log(kp)' if plot_type=="log" else 'kp',
        yaxis_title='log(ki)' if plot_type=="log" else 'ki',
        zaxis_title='log(|kd|)' if plot_type=="log" else 'kd',
    ),
    title=f'3D Heatmap of PID Parameters in Log Space vs {cost_type}'
)

fig.show()