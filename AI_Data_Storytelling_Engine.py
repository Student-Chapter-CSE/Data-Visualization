
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime


def generate_story_data(seed: int = 7) -> dict:
    """Generate a compact but expressive dataset used across scenes."""
    rng = np.random.default_rng(seed)

    # Time series for 24 months with seasonality and trend
    months = pd.date_range('2023-01-01', periods=24, freq='MS')
    t = np.arange(len(months))
    season = 0.35 * np.sin(2 * np.pi * t / 12) + 0.15 * np.cos(2 * np.pi * t / 6)
    trend = 0.02 * t
    noise = rng.normal(0, 0.08, len(t))
    value = (1.0 + season + trend + noise).clip(min=0.25)
    ts = pd.DataFrame({
        'month': months,
        'month_name': months.strftime('%b %Y'),
        'value': value,
        'theta_deg': (months.month - 1) * (360 / 12),
        'radius': 0.5 + 0.5 * (value - value.min()) / (value.max() - value.min())
    })

    # 3D clusters (four clusters in a loose spiral shell)
    num_points = 220
    cluster_ids = rng.integers(0, 4, size=num_points)
    centers = np.array([
        [ 2.5,  0.5,  0.0],
        [-2.0, -1.0,  0.8],
        [ 0.0,  2.3, -0.6],
        [ 1.5, -2.2,  1.2],
    ])
    pts = centers[cluster_ids] + rng.normal(0, 0.6, size=(num_points, 3))
    weights = (rng.random(num_points) ** 2) * 15 + 5
    clusters = pd.DataFrame({
        'x': pts[:, 0],
        'y': pts[:, 1],
        'z': pts[:, 2],
        'cluster': pd.Categorical(cluster_ids),
        'weight': weights
    })

    # Flow/funnel proportions across three stages, varying by segment
    segments = ['Organic', 'Paid', 'Referral', 'Social']
    stage_a = rng.dirichlet(np.ones(len(segments)) * 2.0)
    # Transition matrices
    a_to_b = rng.dirichlet(np.ones(len(segments)) * 1.6, size=len(segments))
    b_to_c = rng.dirichlet(np.ones(len(segments)) * 1.4, size=len(segments))

    # Expand into Sankey nodes and links
    nodes = [f"A:{s}" for s in segments] + [f"B:{s}" for s in segments] + [f"C:{s}" for s in segments]
    node_index = {n: i for i, n in enumerate(nodes)}

    links_src, links_tgt, links_val, links_lbl = [], [], [], []
    total = 1000
    for i, s in enumerate(segments):
        val_a = stage_a[i] * total
        # A -> B
        for j, s2 in enumerate(segments):
            v_ab = val_a * a_to_b[i, j]
            links_src.append(node_index[f"A:{s}"])
            links_tgt.append(node_index[f"B:{s2}"])
            links_val.append(v_ab)
            links_lbl.append(f"{s} → {s2}")
        # B -> C
    # Need intermediate B totals per segment
    b_totals = np.zeros(len(segments))
    for i, s in enumerate(segments):
        val_a = stage_a[i] * total
        b_totals += val_a * a_to_b[i]
    for i, s in enumerate(segments):
        val_b = b_totals[i]
        for j, s3 in enumerate(segments):
            v_bc = val_b * b_to_c[i, j]
            links_src.append(node_index[f"B:{s}"])
            links_tgt.append(node_index[f"C:{s3}"])
            links_val.append(v_bc)
            links_lbl.append(f"{s} → {s3}")

    sankey = {
        'nodes': nodes,
        'links': {
            'source': links_src,
            'target': links_tgt,
            'value': links_val,
            'label': links_lbl,
        }
    }

    return {'time': ts, 'clusters': clusters, 'sankey': sankey, 'segments': segments}


DATA = generate_story_data()


def scene_title(idx: int) -> str:
    return [
        'Scene 1 · Harmonic Time Spiral',
        'Scene 2 · 3D Cluster Bloom',
        'Scene 3 · Flow of Attention',
        'Scene 4 · Morph: Spiral ↔ Bars',
        'Scene 5 · Constellation Radar Glyphs'
    ][idx]


def scene_insights(idx: int) -> list:
    if idx == 0:
        ts = DATA['time']
        peak_row = ts.loc[ts['value'].idxmax()]
        yoy = (ts.loc[12:, 'value'].values - ts.loc[:11, 'value'].values).mean()
        return [
            f"Peak month: {peak_row['month'].strftime('%b %Y')}",
            f"Avg YoY monthly delta (approx): {yoy:+.2f}",
            f"Seasonal amplitude: {(ts['value'].max()-ts['value'].min()):.2f}"
        ]
    if idx == 1:
        cl = DATA['clusters']
        sizes = cl.groupby('cluster')['weight'].sum().sort_values(ascending=False)
        return [
            f"Largest cluster by weight: {sizes.index[0]}",
            f"Cluster spread (x,y,z std): {cl[['x','y','z']].std().mean():.2f}",
            f"Points: {len(cl)}"
        ]
    if idx == 2:
        sank = DATA['sankey']
        total = sum(sank['links']['value'])
        return [
            f"Total flow value: {int(total)}",
            "Two-stage transitions show segment mixing",
            "Downstream distribution highlights stickiest segments"
        ]
    if idx == 3:
        ts = DATA['time']
        return [
            "Use the slider to morph geometry",
            f"Range of values: {ts['value'].min():.2f}–{ts['value'].max():.2f}",
            "Spiral reveals seasonality; bars emphasize absolute deltas"
        ]
    # idx == 4
    ts = DATA['time']
    # Compute simple quarterly features
    q = ts.copy()
    q['quarter'] = q['month'].dt.to_period('Q')
    agg = q.groupby('quarter')['value'].agg(['mean','std']).reset_index()
    return [
        f"Quarters analyzed: {len(agg)}",
        f"Median volatility (std): {agg['std'].median():.2f}",
        "Radar shows balance of stability vs growth"
    ]


def figure_scene(idx: int, morph: float = 0.0) -> go.Figure:
    if idx == 0:
        # Polar spiral
        ts = DATA['time']
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=ts['radius'],
            theta=ts['theta_deg'],
            mode='lines+markers',
            line=dict(color='#7F7EFF', width=3),
            marker=dict(size=6, color=ts['radius'], colorscale='Viridis'),
            name='Seasonal trend'
        ))
        fig.update_layout(
            template='plotly_dark',
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(direction='clockwise')
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            height=520
        )
        return fig

    if idx == 1:
        # 3D clusters
        cl = DATA['clusters']
        fig = go.Figure()
        for key, grp in cl.groupby('cluster'):
            fig.add_trace(go.Scatter3d(
                x=grp['x'], y=grp['y'], z=grp['z'],
                mode='markers',
                marker=dict(size=np.clip(grp['weight']/6, 3, 14), color=key, colorscale='Turbo', opacity=0.85),
                name=f'Cluster {key}'
            ))
        fig.update_layout(
            template='plotly_dark',
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            height=520
        )
        return fig

    if idx == 2:
        # Sankey flow
        sank = DATA['sankey']
        labels = sank['nodes']
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=16,
                line=dict(color='rgba(255,255,255,0.25)', width=1),
                label=labels, color='rgba(127,127,255,0.6)'
            ),
            link=dict(
                source=sank['links']['source'],
                target=sank['links']['target'],
                value=sank['links']['value'],
                label=sank['links']['label']
            )
        )])
        fig.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20), height=520)
        return fig

    if idx == 3:
        # Morph between polar spiral (morph=0) and bar chart (morph=1)
        ts = DATA['time']
        # Interpolate radius to cartesian y via morph
        y_bar = (ts['value'] - ts['value'].min()) / (ts['value'].max() - ts['value'].min())
        r = (1 - morph) * ts['radius'] + morph * (0.5 + 0.5 * y_bar)
        fig = go.Figure()
        if morph < 0.999:
            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=ts['theta_deg'],
                mode='lines+markers',
                line=dict(color='#00E3AE', width=3),
                marker=dict(size=6, color=r, colorscale='Mint'),
                name='Spiral form'
            ))
            fig.update_layout(
                template='plotly_dark',
                polar=dict(radialaxis=dict(visible=False)),
                margin=dict(l=40, r=40, t=40, b=40),
                height=520
            )
        if morph > 0.001:
            # Overlay bar chart projection
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=ts['month_name'], y=y_bar, marker_color='#7F7EFF', name='Bars'))
            fig2.update_layout(template='plotly_dark', xaxis_tickangle=-45)
            # Merge as images for simplicity
            return fig2
        return fig

    # idx == 4: Radar glyphs for quarterly features
    ts = DATA['time']
    q = ts.copy()
    q['quarter'] = q['month'].dt.to_period('Q')
    dfq = q.groupby('quarter')['value'].agg(['mean', 'std']).reset_index()
    # Derived features scaled 0-1
    dfq['growth'] = (dfq['mean'] - dfq['mean'].min()) / (dfq['mean'].max() - dfq['mean'].min() + 1e-9)
    dfq['volatility'] = (dfq['std'] - dfq['std'].min()) / (dfq['std'].max() - dfq['std'].min() + 1e-9)
    dfq['stability'] = 1 - dfq['volatility']
    dfq['momentum'] = dfq['growth'].rolling(2, min_periods=1).mean()
    features = ['growth', 'stability', 'momentum', 'volatility']
    fig = go.Figure()
    for i, row in dfq.iterrows():
        vals = [row[f] for f in features] + [row[features[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=features + [features[0]],
            fill='toself',
            name=str(row['quarter']),
            opacity=0.45
        ))
    fig.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20), height=520, polar=dict(radialaxis=dict(range=[0,1])))
    return fig


app = dash.Dash(__name__)
app.title = '🎭 AI Data Storytelling Engine'


app.layout = html.Div([
    html.Div([
        html.H1('🎭 AI Data Storytelling Engine', style={'margin': '0'}),
        html.P('A narrated, multi-scene tour through your data', style={'opacity': 0.8})
    ], style={'textAlign': 'center', 'padding': '20px 10px'}),

    html.Div([
        html.Button('◀ Previous', id='prev-btn'),
        html.Span(id='scene-title', style={'padding': '0 16px', 'fontWeight': 700}),
        html.Button('Next ▶', id='next-btn')
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '10px'}),

    dcc.Store(id='scene-idx', data=0),

    html.Div([
        html.Label('Morph (Spiral ↔ Bars)', style={'marginRight': '10px'}),
        dcc.Slider(id='morph', min=0, max=1, step=0.01, value=0, tooltip={'always_visible': False})
    ], id='morph-container', style={'maxWidth': 900, 'margin': '12px auto', 'display': 'none'}),

    html.Div([
        dcc.Graph(id='scene-figure')
    ], style={'padding': '10px 10px 0'}),

    html.Div([
        html.Div('Insights', style={'fontWeight': 700, 'marginBottom': '6px'}),
        html.Ul(id='insight-list', style={'margin': 0})
    ], style={'maxWidth': 900, 'margin': '0 auto', 'padding': '10px 20px 30px', 'background': '#111', 'borderRadius': '12px'}),
])


@app.callback(
    Output('scene-idx', 'data'),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    State('scene-idx', 'data'),
    prevent_initial_call=True,
)
def switch_scene(prev_clicks, next_clicks, idx):
    ctx = dash.callback_context
    if not ctx.triggered:
        return idx
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    if trig == 'next-btn':
        return (idx + 1) % 5
    if trig == 'prev-btn':
        return (idx - 1) % 5
    return idx


@app.callback(
    Output('scene-figure', 'figure'),
    Output('scene-title', 'children'),
    Output('insight-list', 'children'),
    Output('morph-container', 'style'),
    Input('scene-idx', 'data'),
    Input('morph', 'value')
)
def render_scene(idx, morph):
    fig = figure_scene(idx, morph or 0.0)
    title = scene_title(idx)
    insights = [html.Li(txt) for txt in scene_insights(idx)]
    morph_style = {'maxWidth': 900, 'margin': '12px auto', 'display': 'block'} if idx == 3 else {'display': 'none'}
    return fig, title, insights, morph_style


if __name__ == '__main__':
    print('🚀 Starting AI Data Storytelling Engine...')
    print('📍 http://127.0.0.1:8060')
    app.run_server(debug=False, host='127.0.0.1', port=8060)


