import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from functools import lru_cache
import os


def generate_data(n_samples: int, noise_sd: float, seed: int, function_type: str = "sine"):
    """Generate synthetic data with different underlying functions."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, size=n_samples))
    
    # Multiple ground truth functions to choose from
    if function_type == "sine":
        y_true = np.sin(x) + 0.3 * np.cos(2 * x) + 0.1 * x
    elif function_type == "polynomial":
        y_true = 0.1 * x**3 - 0.5 * x**2 + 0.3 * x + 0.5
    elif function_type == "step":
        y_true = np.where(x < -1, -1, np.where(x < 1, 0.5 * x, 1))
    elif function_type == "exponential":
        y_true = np.exp(-0.5 * x**2) * np.sin(2 * x)
    else:
        y_true = np.sin(x) + 0.3 * np.cos(2 * x) + 0.1 * x
    
    y = y_true + rng.normal(0, noise_sd, size=n_samples)
    return x.reshape(-1, 1), y, y_true


def fit_poly(x_train, y_train, degree: int, model_type: str = "OLS", alpha: float = 1.0):
    """Fit polynomial regression with different regularization methods."""
    degree = int(max(1, min(20, degree)))
    
    if model_type == "OLS":
        reg = LinearRegression()
    elif model_type == "Ridge":
        reg = Ridge(alpha=float(max(0.01, alpha)))
    elif model_type == "Lasso":
        reg = Lasso(alpha=float(max(0.01, alpha)), max_iter=2000)
    else:
        reg = LinearRegression()
    
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lin", reg)
    ])
    model.fit(x_train, y_train)
    return model


def compute_curves(n_train, n_test, noise_sd, degree, seed, model_type="OLS", alpha=1.0, function_type="sine"):
    """Compute model predictions and errors."""
    x_train, y_train, y_true_train = generate_data(n_train, noise_sd, seed, function_type)
    x_test, y_test, y_true_test = generate_data(n_test, noise_sd, seed + 1, function_type)
    
    model = fit_poly(x_train, y_train, degree, model_type=model_type, alpha=alpha)
    yhat_train = model.predict(x_train)
    yhat_test = model.predict(x_test)
    
    train_mse = mean_squared_error(y_train, yhat_train)
    test_mse = mean_squared_error(y_test, yhat_test)
    
    # For smooth curve visualization
    xs = np.linspace(-3, 3, 300).reshape(-1, 1)
    _, _, ys_true = generate_data(300, 0.0, seed, function_type)
    ys_hat = model.predict(xs)
    
    # Calculate residuals
    train_residuals = y_train - yhat_train
    test_residuals = y_test - yhat_test
    
    return {
        "x_train": x_train[:, 0],
        "y_train": y_train,
        "x_test": x_test[:, 0],
        "y_test": y_test,
        "xs": xs[:, 0],
        "ys_true": ys_true,
        "ys_hat": ys_hat,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_residuals": train_residuals,
        "test_residuals": test_residuals,
        "yhat_train": yhat_train,
        "yhat_test": yhat_test,
    }


@lru_cache(maxsize=512)
def monte_carlo_error(degree: int, noise_sd: float, n_train: int, n_test: int, seed: int, 
                     runs: int, model_type: str, alpha: float, function_type: str):
    """Run Monte Carlo simulation to estimate bias and variance."""
    rng = np.random.default_rng(seed)
    train_mses, test_mses = [], []
    for r in range(runs):
        s = int(rng.integers(0, 10_000))
        res = compute_curves(n_train, n_test, noise_sd, degree, s, model_type=model_type, 
                           alpha=alpha, function_type=function_type)
        train_mses.append(res["train_mse"])
        test_mses.append(res["test_mse"])
    return np.array(train_mses), np.array(test_mses)


def figure_fit(res, show_residuals=False):
    """Create the main fit visualization."""
    fig = go.Figure()
    
    # True function
    fig.add_trace(go.Scatter(
        x=res["xs"], y=res["ys_true"], mode="lines", name="True function",
        line=dict(color="#00E3AE", width=3)
    ))
    
    # Fitted curve
    fig.add_trace(go.Scatter(
        x=res["xs"], y=res["ys_hat"], mode="lines", name=f"Fitted model",
        line=dict(color="#7F7EFF", width=3, dash="dash")
    ))
    
    # Training points
    fig.add_trace(go.Scatter(
        x=res["x_train"], y=res["y_train"], mode="markers", name="Train data",
        marker=dict(size=8, color="#FFD166", line=dict(color="#444", width=0.5))
    ))
    
    # Test points
    fig.add_trace(go.Scatter(
        x=res["x_test"], y=res["y_test"], mode="markers", name="Test data",
        marker=dict(size=7, color="#EF476F", opacity=0.7)
    ))
    
    # Optional: Show residuals as vertical lines
    if show_residuals:
        for i in range(len(res["x_train"])):
            fig.add_trace(go.Scatter(
                x=[res["x_train"][i], res["x_train"][i]],
                y=[res["y_train"][i], res["yhat_train"][i]],
                mode="lines",
                line=dict(color="#FFD166", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            ))
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Input (x)",
        yaxis_title="Output (y)",
        hovermode="closest"
    )
    
    return fig


def figure_error_curve(noise_sd: float, n_train: int, n_test: int, seed: int, 
                      model_type: str, alpha: float, function_type: str):
    """Create the train/test error vs complexity curve."""
    degrees = list(range(1, 16))
    train_means, test_means, test_stds = [], [], []
    
    for d in degrees:
        tr, te = monte_carlo_error(
            d, noise_sd, n_train, n_test, seed, runs=25, 
            model_type=model_type, alpha=alpha, function_type=function_type
        )
        train_means.append(np.mean(tr))
        test_means.append(np.mean(te))
        test_stds.append(np.std(te))
    
    fig = go.Figure()
    
    # Train error
    fig.add_trace(go.Scatter(
        x=degrees, y=train_means, mode="lines+markers", name="Train MSE",
        line=dict(color="#FFD166", width=3),
        marker=dict(size=8)
    ))
    
    # Test error
    fig.add_trace(go.Scatter(
        x=degrees, y=test_means, mode="lines+markers", name="Test MSE",
        line=dict(color="#EF476F", width=3),
        marker=dict(size=8)
    ))
    
    # Error band for test variance
    upper = (np.array(test_means) + np.array(test_stds)).tolist()
    lower = (np.array(test_means) - np.array(test_stds)).tolist()
    
    fig.add_trace(go.Scatter(
        x=degrees + degrees[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(239,71,111,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Test ±1σ"
    ))
    
    # Mark optimal point
    min_idx = np.argmin(test_means)
    fig.add_trace(go.Scatter(
        x=[degrees[min_idx]],
        y=[test_means[min_idx]],
        mode="markers",
        marker=dict(size=15, color="#00E3AE", symbol="star"),
        name="Optimal",
        showlegend=True
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Polynomial Degree (Model Complexity)",
        yaxis_title="Mean Squared Error",
        hovermode="x unified"
    )
    
    return fig


def figure_error_histograms(degree: int, noise_sd: float, n_train: int, n_test: int, 
                           seed: int, model_type: str, alpha: float, function_type: str):
    """Create histograms of train/test errors across multiple runs."""
    tr, te = monte_carlo_error(
        degree, noise_sd, n_train, n_test, seed, runs=80, 
        model_type=model_type, alpha=alpha, function_type=function_type
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=tr, name="Train MSE", opacity=0.75, 
        marker_color="#FFD166", nbinsx=25
    ))
    
    fig.add_trace(go.Histogram(
        x=te, name="Test MSE", opacity=0.6, 
        marker_color="#EF476F", nbinsx=25
    ))
    
    fig.update_layout(
        template="plotly_dark",
        barmode="overlay",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="MSE Value",
        yaxis_title="Frequency",
        hovermode="x"
    )
    
    return fig


def figure_residuals(res):
    """Create residual plot to diagnose model fit."""
    fig = go.Figure()
    
    # Train residuals
    fig.add_trace(go.Scatter(
        x=res["yhat_train"],
        y=res["train_residuals"],
        mode="markers",
        name="Train",
        marker=dict(size=8, color="#FFD166", opacity=0.7)
    ))
    
    # Test residuals
    fig.add_trace(go.Scatter(
        x=res["yhat_test"],
        y=res["test_residuals"],
        mode="markers",
        name="Test",
        marker=dict(size=7, color="#EF476F", opacity=0.7)
    ))
    
    # Zero line
    x_range = [min(res["yhat_train"].min(), res["yhat_test"].min()),
               max(res["yhat_train"].max(), res["yhat_test"].max())]
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[0, 0],
        mode="lines",
        line=dict(color="white", width=1, dash="dash"),
        showlegend=False
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Predicted Value",
        yaxis_title="Residual",
        hovermode="closest"
    )
    
    return fig


# Initialize Dash app
app = dash.Dash(__name__)
app.title = "🎯 Bias-Variance Playground Pro"

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🎯 Bias-Variance Playground Pro", style={"margin": "0", "fontSize": "2.2em"}),
        html.P("Interactive exploration of the bias-variance tradeoff in machine learning",
               style={"opacity": 0.85, "fontSize": "1.1em", "marginTop": "8px"})
    ], style={"textAlign": "center", "padding": "20px 8px", "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "borderRadius": "8px", "margin": "12px"}),

    # Controls Section
    html.Div([
        html.H3("⚙️ Configuration", style={"marginBottom": "16px"}),
        
        # Row 1: Data generation
        html.Div([
            html.Div([
                html.Label("🎲 Random Seed", style={"fontWeight": "bold", "marginBottom": "4px"}),
                dcc.Slider(id="seed", min=0, max=999, step=1, value=42,
                          tooltip={"always_visible": False}, marks={0: "0", 500: "500", 999: "999"})
            ], style={"flex": 1, "minWidth": 180}),
            
            html.Div([
                html.Label("📊 Train Size", style={"fontWeight": "bold", "marginBottom": "4px"}),
                dcc.Slider(id="n_train", min=20, max=400, step=10, value=100,
                          marks={20: "20", 200: "200", 400: "400"})
            ], style={"flex": 1, "minWidth": 180}),
            
            html.Div([
                html.Label("📈 Test Size", style={"fontWeight": "bold", "marginBottom": "4px"}),
                dcc.Slider(id="n_test", min=50, max=600, step=10, value=200,
                          marks={50: "50", 300: "300", 600: "600"})
            ], style={"flex": 1, "minWidth": 180}),
            
            html.Div([
                html.Label("🔊 Noise Level (σ)", style={"fontWeight": "bold", "marginBottom": "4px"}),
                dcc.Slider(id="noise", min=0.0, max=1.5, step=0.05, value=0.25,
                          marks={0: "0", 0.75: "0.75", 1.5: "1.5"})
            ], style={"flex": 1, "minWidth": 180}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px", "flexWrap": "wrap"}),
        
        # Row 2: Model configuration
        html.Div([
            html.Div([
                html.Label("📐 Function Type", style={"fontWeight": "bold", "marginBottom": "8px"}),
                dcc.RadioItems(
                    id="function_type",
                    options=[
                        {"label": "Sine Wave", "value": "sine"},
                        {"label": "Polynomial", "value": "polynomial"},
                        {"label": "Step", "value": "step"},
                        {"label": "Exponential", "value": "exponential"}
                    ],
                    value="sine",
                    inline=False,
                    style={"display": "flex", "flexDirection": "column", "gap": "6px"}
                )
            ], style={"flex": 1, "minWidth": 200}),
            
            html.Div([
                html.Label("🤖 Model Type", style={"fontWeight": "bold", "marginBottom": "8px"}),
                dcc.RadioItems(
                    id="model_type",
                    options=[
                        {"label": "OLS (No Regularization)", "value": "OLS"},
                        {"label": "Ridge (L2)", "value": "Ridge"},
                        {"label": "Lasso (L1)", "value": "Lasso"}
                    ],
                    value="OLS",
                    inline=False,
                    style={"display": "flex", "flexDirection": "column", "gap": "6px"}
                )
            ], style={"flex": 1, "minWidth": 200}),
            
            html.Div([
                html.Label("📏 Polynomial Degree", style={"fontWeight": "bold", "marginBottom": "4px"}),
                dcc.Slider(
                    id="degree",
                    min=1,
                    max=15,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(1, 16, 2)}
                ),
                html.Div(id="degree-display", style={"textAlign": "center", "marginTop": "4px", "fontSize": "1.1em", "color": "#00E3AE"})
            ], style={"flex": 2, "minWidth": 280}),
            
            html.Div([
                html.Label("⚖️ Regularization (α)", style={"fontWeight": "bold", "marginBottom": "4px"}),
                dcc.Slider(
                    id="alpha",
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    value=1.0,
                    marks={0: "0", 5: "5", 10: "10"}
                ),
                html.Div(id="alpha-display", style={"textAlign": "center", "marginTop": "4px", "fontSize": "1.0em"})
            ], style={"flex": 1, "minWidth": 200}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        
    ], style={"padding": "20px", "background": "rgba(255,255,255,0.05)", "borderRadius": "8px", "margin": "12px"}),

    # Main visualization
    html.Div([
        html.H3("📊 Model Fit", style={"marginBottom": "12px"}),
        dcc.Graph(id="fit-graph"),
        html.Div([
            dcc.Checklist(
                id="show-residuals",
                options=[{"label": " Show residual lines", "value": "show"}],
                value=[],
                inline=True,
                style={"fontSize": "0.95em"}
            )
        ], style={"textAlign": "center", "marginTop": "8px"})
    ], style={"padding": "12px", "margin": "12px"}),

    # Error analysis
    html.Div([
        html.Div([
            html.H3("📉 Error vs Complexity", style={"marginBottom": "12px"}),
            dcc.Graph(id="error-curve"),
        ], style={"flex": 1.2, "minWidth": 320}),
        
        html.Div([
            html.H3("📊 Error Distribution", style={"marginBottom": "12px"}),
            dcc.Graph(id="error-hist"),
        ], style={"flex": 1, "minWidth": 320}),
    ], style={"display": "flex", "gap": "16px", "padding": "12px", "flexWrap": "wrap", "margin": "12px"}),

    # Residual plot
    html.Div([
        html.H3("🔍 Residual Analysis", style={"marginBottom": "12px"}),
        dcc.Graph(id="residual-plot"),
        html.P("Well-fitted models show randomly scattered residuals around zero with no patterns.",
               style={"textAlign": "center", "opacity": 0.7, "fontSize": "0.9em", "marginTop": "8px"})
    ], style={"padding": "12px", "margin": "12px"}),

    # Statistics and controls
    html.Div([
        html.Button("🔄 Recompute Bias-Variance", id="mc-btn", n_clicks=0,
                   style={"padding": "12px 24px", "fontSize": "1.1em", "cursor": "pointer",
                          "background": "#667eea", "border": "none", "borderRadius": "6px",
                          "color": "white", "fontWeight": "bold"}),
        html.Div(id="bv-stats", style={"marginTop": "12px", "fontSize": "1.1em", "fontWeight": "bold"})
    ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", 
              "justifyContent": "center", "padding": "20px", "margin": "12px",
              "background": "rgba(255,255,255,0.05)", "borderRadius": "8px"}),

    # Footer
    html.Div([
        html.P(id="line-count", style={"opacity": 0.6, "fontSize": "0.85em"}),
        html.P("💡 Tip: Try increasing polynomial degree to see overfitting. Use Ridge/Lasso to reduce it!",
               style={"opacity": 0.7, "fontSize": "0.9em", "marginTop": "8px"})
    ], style={"textAlign": "center", "padding": "16px"}),
])


@app.callback(
    Output("fit-graph", "figure"),
    Output("error-curve", "figure"),
    Output("error-hist", "figure"),
    Output("residual-plot", "figure"),
    Output("bv-stats", "children"),
    Output("line-count", "children"),
    Output("degree-display", "children"),
    Output("alpha-display", "children"),
    Input("n_train", "value"),
    Input("n_test", "value"),
    Input("noise", "value"),
    Input("degree", "value"),
    Input("seed", "value"),
    Input("model_type", "value"),
    Input("alpha", "value"),
    Input("function_type", "value"),
    Input("show-residuals", "value"),
    Input("mc-btn", "n_clicks"),
)
def update_all(n_train, n_test, noise_sd, degree, seed, model_type, alpha, 
               function_type, show_residuals, mc_clicks):
    # Validate and clamp inputs
    n_train = int(max(10, min(1000, n_train or 100)))
    n_test = int(max(20, min(2000, n_test or 200)))
    noise_sd = float(max(0.0, min(2.0, noise_sd or 0.25)))
    degree = int(max(1, min(20, degree or 5)))
    alpha = float(max(0.0, min(100.0, alpha or 1.0)))
    seed = int(seed or 42)
    
    # Compute results
    res = compute_curves(n_train, n_test, noise_sd, degree, seed, 
                        model_type=model_type, alpha=alpha, function_type=function_type)
    
    # Create figures
    show_res_lines = "show" in (show_residuals or [])
    fig_fit = figure_fit(res, show_residuals=show_res_lines)
    fig_err_curve = figure_error_curve(noise_sd, n_train, n_test, seed, 
                                       model_type, alpha, function_type)
    fig_hist = figure_error_histograms(degree, noise_sd, n_train, n_test, seed,
                                       model_type, alpha, function_type)
    fig_residual = figure_residuals(res)
    
    # Bias-variance decomposition
    ctx = callback_context
    if mc_clicks and mc_clicks > 0:
        runs = 50
        xs = res["xs"].reshape(-1, 1)
        preds = []
        rng = np.random.default_rng(seed)
        
        for _ in range(runs):
            s = int(rng.integers(0, 10_000))
            x_tr, y_tr, _ = generate_data(n_train, noise_sd, s, function_type)
            m = fit_poly(x_tr, y_tr, degree, model_type=model_type, alpha=alpha)
            preds.append(m.predict(xs))
        
        P = np.stack(preds, axis=0)
        mean_pred = P.mean(axis=0)
        var_pred = P.var(axis=0)
        bias_sq = (mean_pred - res["ys_true"]) ** 2
        noise_var = noise_sd ** 2
        
        bv_text = html.Div([
            html.Span(f"Bias² = {bias_sq.mean():.4f}", style={"color": "#00E3AE", "marginRight": "20px"}),
            html.Span(f"Variance = {var_pred.mean():.4f}", style={"color": "#7F7EFF", "marginRight": "20px"}),
            html.Span(f"Noise ≈ {noise_var:.4f}", style={"color": "#FFD166", "marginRight": "20px"}),
            html.Span(f"Total ≈ {bias_sq.mean() + var_pred.mean() + noise_var:.4f}", 
                     style={"color": "#EF476F"})
        ])
    else:
        bv_text = "Click the button above to compute bias-variance decomposition"
    
    # Line count
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)
        line_text = f"📝 This application contains {num_lines} lines of Python code"
    except Exception:
        line_text = "📝 Enhanced Bias-Variance Playground"
    
    # Display values
    degree_display = f"Degree: {degree} {'(High Complexity ⚠️)' if degree > 10 else ''}"
    alpha_display = f"α = {alpha:.1f} {'(Disabled for OLS)' if model_type == 'OLS' else ''}"
    
    return fig_fit, fig_err_curve, fig_hist, fig_residual, bv_text, line_text, degree_display, alpha_display


@app.callback(
    Output("alpha", "disabled"),
    Input("model_type", "value")
)
def toggle_alpha(model_type):
    return model_type == "OLS"


if __name__ == "__main__":
    print("🚀 Starting Enhanced Bias-Variance Playground...")
    print("📍 Open your browser to: http://127.0.0.1:8070")
    print("💡 Explore bias-variance tradeoff interactively!")
    app.run_server(debug=False, host="127.0.0.1", port=8070)