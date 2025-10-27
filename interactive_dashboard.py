#!/usr/bin/env python3
"""
🎨 Interactive Data Visualization Dashboard
==========================================

A comprehensive, interactive dashboard showcasing various data visualization techniques
using Plotly Dash. This dashboard combines multiple data sources and visualization types
to create an engaging, educational experience.

Features:
- Multi-tab interface with different visualization categories
- Interactive controls and filters
- Real-time data updates
- Responsive design
- Custom themes and styling
- Educational tooltips and explanations

Author: Data Visualization Collection
Date: 2024
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table, State, clientside_callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import base64
import io
import csv
from dash.exceptions import PreventUpdate

# Set random seed for reproducible data
np.random.seed(42)
random.seed(42)

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "🎨 Interactive Data Visualization Dashboard"

# Disable dev tools to avoid compatibility issues with newer Python versions
app.enable_dev_tools(dev_tools_ui=False, dev_tools_hot_reload=False)

# Global state for theme and filters
current_theme = "light"
global_filters = {
    'date_range': None,
    'category_filter': 'All',
    'region_filter': 'All'
}

# Enhanced CSS styling with theme support
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --bg-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --bg-secondary: rgba(255, 255, 255, 0.95);
                --text-primary: #333;
                --text-secondary: #666;
                --accent-color: #667eea;
                --card-bg: white;
                --shadow: 0 20px 40px rgba(0,0,0,0.1);
                --border-radius: 15px;
            }
            
            [data-theme="dark"] {
                --bg-primary: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                --bg-secondary: rgba(44, 62, 80, 0.95);
                --text-primary: #ecf0f1;
                --text-secondary: #bdc3c7;
                --accent-color: #3498db;
                --card-bg: #34495e;
                --shadow: 0 20px 40px rgba(0,0,0,0.3);
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: var(--bg-primary);
                margin: 0;
                padding: 0;
                color: var(--text-primary);
                transition: all 0.3s ease;
            }
            
            .main-container {
                background: var(--bg-secondary);
                border-radius: var(--border-radius);
                margin: 20px;
                box-shadow: var(--shadow);
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .header {
                background: var(--bg-primary);
                color: white;
                padding: 30px;
                text-align: center;
                position: relative;
            }
            
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .theme-toggle {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.2);
                border: none;
                border-radius: 50px;
                padding: 10px 20px;
                color: white;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            
            .theme-toggle:hover {
                background: rgba(255,255,255,0.3);
                transform: scale(1.05);
            }
            
            .filters-bar {
                background: var(--card-bg);
                padding: 20px;
                border-bottom: 1px solid #eee;
                display: flex;
                gap: 20px;
                align-items: center;
                flex-wrap: wrap;
                transition: all 0.3s ease;
            }
            
            .filter-item {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            
            .filter-item label {
                font-size: 12px;
                color: var(--text-secondary);
                font-weight: 600;
            }
            
            .kpi-row {
                display: flex;
                gap: 20px;
                padding: 20px;
                background: var(--card-bg);
                border-bottom: 1px solid #eee;
                flex-wrap: wrap;
                transition: all 0.3s ease;
            }
            
            .kpi-card {
                flex: 1;
                min-width: 200px;
                background: var(--card-bg);
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border-left: 4px solid var(--accent-color);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }
            
            .kpi-value {
                font-size: 2.5em;
                font-weight: bold;
                color: var(--accent-color);
                margin: 0;
                transition: all 0.3s ease;
            }
            
            .kpi-label {
                color: var(--text-secondary);
                margin: 5px 0 0 0;
                font-size: 0.9em;
                font-weight: 600;
            }
            
            .kpi-change {
                font-size: 0.8em;
                margin-top: 5px;
                font-weight: 600;
            }
            
            .kpi-change.positive {
                color: #27ae60;
            }
            
            .kpi-change.negative {
                color: #e74c3c;
            }
            
            .tab-content {
                padding: 30px;
                transition: all 0.3s ease;
            }
            
            .metric-card {
                background: var(--card-bg);
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border-left: 4px solid var(--accent-color);
                transition: all 0.3s ease;
            }
            
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: var(--accent-color);
                margin: 0;
            }
            
            .metric-label {
                color: var(--text-secondary);
                margin: 5px 0 0 0;
                font-size: 0.9em;
            }
            
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #27ae60;
                color: white;
                padding: 15px 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                z-index: 1000;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .export-buttons {
                position: fixed;
                bottom: 20px;
                right: 20px;
                display: flex;
                gap: 10px;
                z-index: 1000;
            }
            
            .export-btn {
                background: var(--accent-color);
                color: white;
                border: none;
                border-radius: 50px;
                padding: 12px 20px;
                cursor: pointer;
                font-size: 14px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            
            .export-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            }
            
            .ai-insights {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .ai-insights h4 {
                margin: 0 0 15px 0;
                font-size: 1.2em;
            }
            
            .ai-insights ul {
                margin: 0;
                padding-left: 20px;
            }
            
            .ai-insights li {
                margin: 8px 0;
                line-height: 1.4;
            }
            
            .drilldown-table {
                margin-top: 20px;
                background: var(--card-bg);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
            }
            
            .fade-in {
                animation: fadeIn 0.5s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def generate_sales_data():
    """Generate realistic sales data for e-commerce dashboard"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate seasonal sales pattern
    base_sales = 1000
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    trend = np.linspace(1, 1.2, len(dates))
    noise = np.random.normal(0, 0.1, len(dates))
    
    sales = base_sales * seasonal_factor * trend * (1 + noise)
    sales = np.maximum(sales, 0)  # Ensure non-negative sales
    
    return pd.DataFrame({
        'date': dates,
        'sales': sales,
        'month': dates.month_name(),
        'quarter': dates.quarter,
        'day_of_week': dates.day_name()
    })

def generate_customer_data():
    """Generate customer demographics and behavior data"""
    np.random.seed(42)
    n_customers = 1000
    
    # Age distribution (skewed towards younger adults)
    ages = np.random.gamma(2, 15, n_customers) + 18
    ages = np.clip(ages, 18, 80)
    
    # Gender distribution
    genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.45, 0.50, 0.05])
    
    # Income (correlated with age)
    income = ages * 1000 + np.random.normal(0, 5000, n_customers)
    income = np.maximum(income, 20000)
    
    # Purchase frequency (inversely correlated with income)
    purchase_freq = np.random.poisson(lam=20 - (income - 30000) / 10000, size=n_customers)
    purchase_freq = np.maximum(purchase_freq, 1)
    
    # Customer satisfaction (0-10 scale)
    satisfaction = np.random.beta(2, 1, n_customers) * 10
    
    return pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': ages,
        'gender': genders,
        'income': income,
        'purchase_frequency': purchase_freq,
        'satisfaction': satisfaction,
        'lifetime_value': income * 0.1 + np.random.normal(0, 1000, n_customers)
    })

def generate_product_data():
    """Generate product performance data"""
    np.random.seed(42)
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
    n_products = 50
    
    products = []
    for i in range(n_products):
        category = np.random.choice(categories)
        
        # Price based on category
        base_prices = {'Electronics': 200, 'Clothing': 50, 'Books': 20, 
                      'Home & Garden': 80, 'Sports': 100, 'Beauty': 30}
        price = base_prices[category] * np.random.uniform(0.5, 2.0)
        
        # Sales volume (inversely related to price)
        sales_volume = int(1000 / (price / 50) * np.random.uniform(0.5, 1.5))
        
        # Rating (slightly correlated with price)
        rating = min(5, max(1, 3 + (price / 100) * 0.5 + np.random.normal(0, 0.5)))
        
        products.append({
            'product_id': i + 1,
            'name': f'Product {i+1}',
            'category': category,
            'price': round(price, 2),
            'sales_volume': sales_volume,
            'rating': round(rating, 1),
            'revenue': round(price * sales_volume, 2)
        })
    
    return pd.DataFrame(products)

def generate_geographic_data():
    """Generate geographic sales data"""
    np.random.seed(42)
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil', 'India', 'China']
    
    # GDP per capita (rough estimates)
    gdp_per_capita = [65000, 45000, 42000, 48000, 40000, 39000, 55000, 15000, 2000, 10000]
    
    # Population (in millions)
    population = [330, 38, 67, 83, 67, 125, 25, 213, 1380, 1400]
    
    # Sales correlated with GDP and population
    sales = []
    for i, country in enumerate(countries):
        base_sales = (gdp_per_capita[i] / 1000) * (population[i] / 100) * np.random.uniform(0.8, 1.2)
        sales.append(max(0, base_sales))
    
    return pd.DataFrame({
        'country': countries,
        'sales': sales,
        'gdp_per_capita': gdp_per_capita,
        'population': population,
        'region': ['North America', 'North America', 'Europe', 'Europe', 'Europe', 
                  'Asia', 'Oceania', 'South America', 'Asia', 'Asia']
    })

# Generate all datasets
sales_df = generate_sales_data()
customer_df = generate_customer_data()
product_df = generate_product_data()
geo_df = generate_geographic_data()

# =============================================================================
# DASHBOARD LAYOUT
# =============================================================================

app.layout = html.Div([
    # Header with theme toggle
    html.Div([
        html.H1("🎨 Interactive Data Visualization Dashboard", className="header"),
        html.P("Explore data through interactive visualizations and gain insights", 
               style={'margin': '10px 0 0 0', 'fontSize': '1.2em', 'opacity': '0.9'}),
        html.Button("🌙 Dark Mode", id="theme-toggle", className="theme-toggle")
    ], className="header"),
    
    # Global Filters Bar
    html.Div([
        html.Div([
            html.Label("📅 Date Range"),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=sales_df['date'].min(),
                end_date=sales_df['date'].max(),
                display_format='YYYY-MM-DD'
            )
        ], className="filter-item"),
        
        html.Div([
            html.Label("🏷️ Category"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': 'All Categories', 'value': 'All'}] + 
                       [{'label': cat, 'value': cat} for cat in product_df['category'].unique()],
                value='All',
                clearable=False
            )
        ], className="filter-item"),
        
        html.Div([
            html.Label("🌍 Region"),
            dcc.Dropdown(
                id='region-filter',
                options=[{'label': 'All Regions', 'value': 'All'}] + 
                       [{'label': region, 'value': region} for region in geo_df['region'].unique()],
                value='All',
                clearable=False
            )
        ], className="filter-item"),
        
        html.Div([
            html.Label("🔄 Live Updates"),
            html.Div([
                dcc.Interval(
                    id='interval-component',
                    interval=5000,  # Update every 5 seconds
                    n_intervals=0
                ),
                html.Span("ON", id="live-status", style={'color': '#27ae60', 'fontWeight': 'bold'})
            ])
        ], className="filter-item")
    ], className="filters-bar"),
    
    # KPI Cards Row
    html.Div([
        html.Div([
            html.H3(id="kpi-revenue", className="kpi-value"),
            html.P("Total Revenue", className="kpi-label"),
            html.P(id="kpi-revenue-change", className="kpi-change")
        ], className="kpi-card"),
        
        html.Div([
            html.H3(id="kpi-growth", className="kpi-value"),
            html.P("Growth Rate", className="kpi-label"),
            html.P(id="kpi-growth-change", className="kpi-change")
        ], className="kpi-card"),
        
        html.Div([
            html.H3(id="kpi-customers", className="kpi-value"),
            html.P("Active Customers", className="kpi-label"),
            html.P(id="kpi-customers-change", className="kpi-change")
        ], className="kpi-card"),
        
        html.Div([
            html.H3(id="kpi-retention", className="kpi-value"),
            html.P("Retention Rate", className="kpi-label"),
            html.P(id="kpi-retention-change", className="kpi-change")
        ], className="kpi-card")
    ], className="kpi-row"),
    
    # Main content
    html.Div([
        dcc.Tabs(id="main-tabs", value="overview", children=[
            # Overview Tab
            dcc.Tab(label="📊 Overview", value="overview", className="custom-tab"),
            dcc.Tab(label="📈 Sales Analytics", value="sales", className="custom-tab"),
            dcc.Tab(label="👥 Customer Insights", value="customers", className="custom-tab"),
            dcc.Tab(label="🛍️ Product Performance", value="products", className="custom-tab"),
            dcc.Tab(label="🌍 Geographic Analysis", value="geographic", className="custom-tab"),
            dcc.Tab(label="🔮 Predictive Analytics", value="predictive", className="custom-tab")
        ]),
        
        html.Div(id="tab-content", className="tab-content"),
        
        # Drilldown Table
        html.Div(id="drilldown-table", className="drilldown-table", style={'display': 'none'})
    ], className="main-container"),
    
    # Export Buttons
    html.Div([
        html.Button("📊 Export CSV", id="export-csv-btn", className="export-btn"),
        html.Button("📈 Export Charts", id="export-charts-btn", className="export-btn"),
        html.Button("📋 Export Report", id="export-report-btn", className="export-btn")
    ], className="export-buttons"),
    
    # Notification Container
    html.Div(id="notification-container"),
    
    # Hidden divs for storing data
    dcc.Store(id='filtered-data-store'),
    dcc.Store(id='theme-store', data='light'),
    dcc.Store(id='drilldown-data-store')
])

# =============================================================================
# CALLBACK FUNCTIONS
# =============================================================================

# Theme toggle callback
@app.callback(
    [Output("theme-store", "data"),
     Output("theme-toggle", "children")],
    [Input("theme-toggle", "n_clicks")],
    [State("theme-store", "data")]
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks is None:
        return "light", "🌙 Dark Mode"
    
    new_theme = "dark" if current_theme == "light" else "light"
    button_text = "☀️ Light Mode" if new_theme == "dark" else "🌙 Dark Mode"
    
    return new_theme, button_text

# Global filters callback
@app.callback(
    Output("filtered-data-store", "data"),
    [Input("date-range-picker", "start_date"),
     Input("date-range-picker", "end_date"),
     Input("category-filter", "value"),
     Input("region-filter", "value"),
     Input("interval-component", "n_intervals")]
)
def update_filtered_data(start_date, end_date, category, region, n_intervals):
    """Update filtered data based on global filters"""
    
    # Apply date filter
    filtered_sales = sales_df.copy()
    if start_date and end_date:
        filtered_sales = filtered_sales[
            (filtered_sales['date'] >= start_date) & 
            (filtered_sales['date'] <= end_date)
        ]
    
    # Apply category filter
    filtered_products = product_df.copy()
    if category != 'All':
        filtered_products = filtered_products[filtered_products['category'] == category]
    
    # Apply region filter
    filtered_geo = geo_df.copy()
    if region != 'All':
        filtered_geo = filtered_geo[filtered_geo['region'] == region]
    
    # Simulate live data updates
    if n_intervals > 0:
        # Add some random variation to simulate real-time updates
        noise_factor = 1 + np.random.normal(0, 0.02, len(filtered_sales))
        filtered_sales['sales'] = filtered_sales['sales'] * noise_factor
    
    return {
        'sales': filtered_sales.to_dict('records'),
        'products': filtered_products.to_dict('records'),
        'customers': customer_df.to_dict('records'),
        'geo': filtered_geo.to_dict('records')
    }

# KPI cards callback
@app.callback(
    [Output("kpi-revenue", "children"),
     Output("kpi-revenue-change", "children"),
     Output("kpi-revenue-change", "className"),
     Output("kpi-growth", "children"),
     Output("kpi-growth-change", "children"),
     Output("kpi-growth-change", "className"),
     Output("kpi-customers", "children"),
     Output("kpi-customers-change", "children"),
     Output("kpi-customers-change", "className"),
     Output("kpi-retention", "children"),
     Output("kpi-retention-change", "children"),
     Output("kpi-retention-change", "className")],
    [Input("filtered-data-store", "data")]
)
def update_kpi_cards(filtered_data):
    """Update KPI cards with filtered data"""
    
    if not filtered_data:
        return ["$0", "0%", "kpi-change", "0%", "0%", "kpi-change", 
                "0", "0", "kpi-change", "0%", "0%", "kpi-change"]
    
    sales_data = pd.DataFrame(filtered_data['sales'])
    products_data = pd.DataFrame(filtered_data['products'])
    customers_data = pd.DataFrame(filtered_data['customers'])
    
    # Calculate KPIs
    total_revenue = sales_data['sales'].sum() if not sales_data.empty else 0
    revenue_change = np.random.uniform(5, 15)  # Simulated growth
    
    growth_rate = np.random.uniform(8, 20)  # Simulated growth rate
    growth_change = np.random.uniform(-2, 5)
    
    active_customers = len(customers_data)
    customers_change = np.random.uniform(2, 8)
    
    retention_rate = np.random.uniform(75, 95)
    retention_change = np.random.uniform(-1, 3)
    
    return [
        f"${total_revenue:,.0f}",
        f"+{revenue_change:.1f}%",
        "kpi-change positive",
        f"{growth_rate:.1f}%",
        f"{'+' if growth_change > 0 else ''}{growth_change:.1f}%",
        f"kpi-change {'positive' if growth_change > 0 else 'negative'}",
        f"{active_customers:,}",
        f"+{customers_change:.1f}%",
        "kpi-change positive",
        f"{retention_rate:.1f}%",
        f"{'+' if retention_change > 0 else ''}{retention_change:.1f}%",
        f"kpi-change {'positive' if retention_change > 0 else 'negative'}"
    ]

# Main tab content callback
@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "value"),
     Input("filtered-data-store", "data")]
)
def render_tab_content(active_tab, filtered_data):
    """Render content based on selected tab and filtered data"""
    
    if not filtered_data:
        return html.Div("Loading...", style={'textAlign': 'center', 'padding': '50px'})
    
    if active_tab == "overview":
        return create_overview_tab(filtered_data)
    elif active_tab == "sales":
        return create_sales_tab(filtered_data)
    elif active_tab == "customers":
        return create_customers_tab(filtered_data)
    elif active_tab == "products":
        return create_products_tab(filtered_data)
    elif active_tab == "geographic":
        return create_geographic_tab(filtered_data)
    elif active_tab == "predictive":
        return create_predictive_tab(filtered_data)

# Export functionality
@app.callback(
    Output("notification-container", "children"),
    [Input("export-csv-btn", "n_clicks"),
     Input("export-charts-btn", "n_clicks"),
     Input("export-report-btn", "n_clicks")],
    [State("filtered-data-store", "data")]
)
def handle_exports(csv_clicks, charts_clicks, report_clicks, filtered_data):
    """Handle export functionality"""
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "export-csv-btn" and csv_clicks:
        return html.Div("📊 CSV data exported successfully!", className="notification")
    elif button_id == "export-charts-btn" and charts_clicks:
        return html.Div("📈 Charts exported as images!", className="notification")
    elif button_id == "export-report-btn" and report_clicks:
        return html.Div("📋 Report generated and downloaded!", className="notification")
    
    return ""

def generate_ai_insights(sales_data, products_data, customers_data):
    """Generate AI-powered insights based on the data"""
    insights = []
    
    if not sales_data.empty:
        # Sales insights
        peak_month = sales_data.groupby('month')['sales'].sum().idxmax()
        total_sales = sales_data['sales'].sum()
        avg_daily = sales_data['sales'].mean()
        
        insights.append(f"📈 Sales peaked in {peak_month} with ${total_sales:,.0f} total revenue")
        insights.append(f"💰 Average daily sales: ${avg_daily:,.0f}")
        
        # Growth trend
        if len(sales_data) > 30:
            recent_avg = sales_data.tail(30)['sales'].mean()
            earlier_avg = sales_data.head(30)['sales'].mean()
            growth = ((recent_avg - earlier_avg) / earlier_avg) * 100
            insights.append(f"📊 Sales trend: {'+' if growth > 0 else ''}{growth:.1f}% change over time")
    
    if not customers_data.empty:
        # Customer insights
        avg_satisfaction = customers_data['satisfaction'].mean()
        high_satisfaction = len(customers_data[customers_data['satisfaction'] >= 8])
        insights.append(f"😊 Customer satisfaction: {avg_satisfaction:.1f}/10 ({high_satisfaction} highly satisfied customers)")
        
        # Demographics
        avg_age = customers_data['age'].mean()
        insights.append(f"👥 Average customer age: {avg_age:.0f} years")
    
    if not products_data.empty:
        # Product insights
        top_category = products_data.groupby('category')['revenue'].sum().idxmax()
        avg_rating = products_data['rating'].mean()
        insights.append(f"🏆 Top category: {top_category} (avg rating: {avg_rating:.1f}/5)")
        
        # Pricing insights
        avg_price = products_data['price'].mean()
        insights.append(f"💵 Average product price: ${avg_price:.0f}")
    
    return insights[:5]  # Return top 5 insights

def create_overview_tab(filtered_data):
    """Create the overview dashboard with key metrics and AI insights"""
    
    sales_data = pd.DataFrame(filtered_data['sales'])
    products_data = pd.DataFrame(filtered_data['products'])
    customers_data = pd.DataFrame(filtered_data['customers'])
    
    # Calculate key metrics
    total_sales = sales_data['sales'].sum() if not sales_data.empty else 0
    avg_daily_sales = sales_data['sales'].mean() if not sales_data.empty else 0
    total_customers = len(customers_data)
    avg_satisfaction = customers_data['satisfaction'].mean() if not customers_data.empty else 0
    total_products = len(products_data)
    top_product = products_data.loc[products_data['revenue'].idxmax(), 'name'] if not products_data.empty else "N/A"
    
    # AI-powered insights
    ai_insights = generate_ai_insights(sales_data, products_data, customers_data)
    
    # Sales trend chart
    if not sales_data.empty:
        sales_trend = px.line(sales_data, x='date', y='sales', 
                             title="Sales Trend Over Time",
                             labels={'sales': 'Sales ($)', 'date': 'Date'})
        sales_trend.update_layout(height=400, showlegend=False)
    else:
        sales_trend = go.Figure()
        sales_trend.add_annotation(text="No data available for selected filters", 
                                  xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        sales_trend.update_layout(height=400, title="Sales Trend Over Time")
    
    # Customer satisfaction distribution
    if not customers_data.empty:
        satisfaction_hist = px.histogram(customers_data, x='satisfaction', nbins=20,
                                       title="Customer Satisfaction Distribution",
                                       labels={'satisfaction': 'Satisfaction Score', 'count': 'Number of Customers'})
        satisfaction_hist.update_layout(height=400)
    else:
        satisfaction_hist = go.Figure()
        satisfaction_hist.add_annotation(text="No data available", 
                                       xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        satisfaction_hist.update_layout(height=400, title="Customer Satisfaction Distribution")
    
    # Product revenue pie chart
    if not products_data.empty:
        category_revenue = products_data.groupby('category')['revenue'].sum().reset_index()
        revenue_pie = px.pie(category_revenue, values='revenue', names='category',
                            title="Revenue by Product Category")
        revenue_pie.update_layout(height=400)
    else:
        revenue_pie = go.Figure()
        revenue_pie.add_annotation(text="No data available", 
                                  xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        revenue_pie.update_layout(height=400, title="Revenue by Product Category")
    
    return html.Div([
        # AI Insights Panel
        html.Div([
            html.H4("🤖 AI-Powered Insights"),
            html.Ul([
                html.Li(insight) for insight in ai_insights
            ])
        ], className="ai-insights"),
        
        html.Div([
            html.Div([dcc.Graph(figure=sales_trend)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=satisfaction_hist)], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=revenue_pie)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.H4("🎯 Key Metrics Summary"),
                html.Ul([
                    html.Li(f"Total Revenue: ${total_sales:,.0f}"),
                    html.Li(f"Average Daily Sales: ${avg_daily_sales:,.0f}"),
                    html.Li(f"Total Customers: {total_customers:,}"),
                    html.Li(f"Average Satisfaction: {avg_satisfaction:.1f}/10"),
                    html.Li(f"Active Products: {total_products}"),
                    html.Li(f"Top Product: {top_product}")
                ])
            ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px'})
        ])
    ], className="fade-in")

def create_sales_tab(filtered_data):
    """Create the sales analytics tab with drilldown functionality"""
    
    sales_data = pd.DataFrame(filtered_data['sales'])
    
    if sales_data.empty:
        return html.Div("No sales data available for selected filters", 
                       style={'textAlign': 'center', 'padding': '50px'})
    
    # Sales by month
    monthly_sales = sales_data.groupby('month')['sales'].sum().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sales['month'] = pd.Categorical(monthly_sales['month'], categories=month_order, ordered=True)
    monthly_sales = monthly_sales.sort_values('month')
    
    monthly_bar = px.bar(monthly_sales, x='month', y='sales',
                        title="Monthly Sales Performance (Click to drill down)",
                        labels={'sales': 'Sales ($)', 'month': 'Month'})
    monthly_bar.update_layout(height=400, xaxis_tickangle=-45)
    
    # Sales by day of week
    daily_sales = sales_data.groupby('day_of_week')['sales'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sales['day_of_week'] = pd.Categorical(daily_sales['day_of_week'], categories=day_order, ordered=True)
    daily_sales = daily_sales.sort_values('day_of_week')
    
    daily_bar = px.bar(daily_sales, x='day_of_week', y='sales',
                      title="Average Sales by Day of Week",
                      labels={'sales': 'Avg Sales ($)', 'day_of_week': 'Day'})
    daily_bar.update_layout(height=400, xaxis_tickangle=-45)
    
    # Sales trend with moving average
    sales_data['ma_7'] = sales_data['sales'].rolling(window=7).mean()
    sales_data['ma_30'] = sales_data['sales'].rolling(window=30).mean()
    
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=sales_data['date'], y=sales_data['sales'], 
                                  name='Daily Sales', opacity=0.6, line=dict(color='blue')))
    trend_fig.add_trace(go.Scatter(x=sales_data['date'], y=sales_data['ma_7'], 
                                  name='7-Day Moving Average', line=dict(color='red')))
    trend_fig.add_trace(go.Scatter(x=sales_data['date'], y=sales_data['ma_30'], 
                                  name='30-Day Moving Average', line=dict(color='green')))
    trend_fig.update_layout(title="Sales Trend with Moving Averages", 
                           xaxis_title="Date", yaxis_title="Sales ($)", height=400)
    
    # Quarterly comparison
    quarterly_sales = sales_data.groupby('quarter')['sales'].sum().reset_index()
    quarterly_pie = px.pie(quarterly_sales, values='sales', names='quarter',
                          title="Sales Distribution by Quarter")
    quarterly_pie.update_layout(height=400)
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=monthly_bar)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=daily_bar)], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=trend_fig)], style={'width': '70%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=quarterly_pie)], style={'width': '30%', 'display': 'inline-block'})
        ])
    ], className="fade-in")

def create_customers_tab():
    """Create the customer insights tab"""
    
    # Age distribution
    age_hist = px.histogram(customer_df, x='age', nbins=20,
                           title="Customer Age Distribution",
                           labels={'age': 'Age', 'count': 'Number of Customers'})
    age_hist.update_layout(height=400)
    
    # Income vs Satisfaction scatter
    income_satisfaction = px.scatter(customer_df, x='income', y='satisfaction', 
                                   color='gender', size='purchase_frequency',
                                   title="Income vs Satisfaction by Gender",
                                   labels={'income': 'Income ($)', 'satisfaction': 'Satisfaction Score'})
    income_satisfaction.update_layout(height=400)
    
    # Gender distribution
    gender_counts = customer_df['gender'].value_counts().reset_index()
    gender_counts.columns = ['gender', 'count']
    gender_pie = px.pie(gender_counts, values='count', names='gender',
                       title="Customer Gender Distribution")
    gender_pie.update_layout(height=400)
    
    # Purchase frequency by age group
    customer_df['age_group'] = pd.cut(customer_df['age'], 
                                     bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    age_group_purchases = customer_df.groupby('age_group')['purchase_frequency'].mean().reset_index()
    
    age_group_bar = px.bar(age_group_purchases, x='age_group', y='purchase_frequency',
                          title="Average Purchase Frequency by Age Group",
                          labels={'purchase_frequency': 'Avg Purchases', 'age_group': 'Age Group'})
    age_group_bar.update_layout(height=400)
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=age_hist)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=gender_pie)], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=income_satisfaction)], style={'width': '60%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=age_group_bar)], style={'width': '40%', 'display': 'inline-block'})
        ])
    ])

def create_products_tab():
    """Create the product performance tab"""
    
    # Price vs Sales Volume scatter
    price_volume = px.scatter(product_df, x='price', y='sales_volume', 
                             color='category', size='revenue',
                             title="Price vs Sales Volume by Category",
                             labels={'price': 'Price ($)', 'sales_volume': 'Sales Volume'})
    price_volume.update_layout(height=400)
    
    # Revenue by category
    category_revenue = product_df.groupby('category')['revenue'].sum().reset_index()
    category_bar = px.bar(category_revenue, x='category', y='revenue',
                         title="Total Revenue by Category",
                         labels={'revenue': 'Revenue ($)', 'category': 'Category'})
    category_bar.update_layout(height=400, xaxis_tickangle=-45)
    
    # Rating distribution
    rating_hist = px.histogram(product_df, x='rating', nbins=10,
                              title="Product Rating Distribution",
                              labels={'rating': 'Rating', 'count': 'Number of Products'})
    rating_hist.update_layout(height=400)
    
    # Top 10 products by revenue
    top_products = product_df.nlargest(10, 'revenue')
    top_products_bar = px.bar(top_products, x='name', y='revenue',
                             title="Top 10 Products by Revenue",
                             labels={'revenue': 'Revenue ($)', 'name': 'Product'})
    top_products_bar.update_layout(height=400, xaxis_tickangle=-45)
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=price_volume)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=category_bar)], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=rating_hist)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=top_products_bar)], style={'width': '50%', 'display': 'inline-block'})
        ])
    ])

def create_geographic_tab():
    """Create the geographic analysis tab"""
    
    # World map visualization
    world_map = px.choropleth(geo_df, 
                             locations='country',
                             locationmode='country names',
                             color='sales',
                             title="Sales by Country",
                             color_continuous_scale='Viridis')
    world_map.update_layout(height=500)
    
    # Sales by region
    region_sales = geo_df.groupby('region')['sales'].sum().reset_index()
    region_pie = px.pie(region_sales, values='sales', names='region',
                       title="Sales Distribution by Region")
    region_pie.update_layout(height=400)
    
    # GDP vs Sales scatter
    gdp_sales = px.scatter(geo_df, x='gdp_per_capita', y='sales', 
                          size='population', color='region',
                          title="GDP per Capita vs Sales",
                          labels={'gdp_per_capita': 'GDP per Capita ($)', 'sales': 'Sales ($)'})
    gdp_sales.update_layout(height=400)
    
    # Top countries bar chart
    top_countries = geo_df.nlargest(8, 'sales')
    countries_bar = px.bar(top_countries, x='country', y='sales',
                          title="Top Countries by Sales",
                          labels={'sales': 'Sales ($)', 'country': 'Country'})
    countries_bar.update_layout(height=400, xaxis_tickangle=-45)
    
    return html.Div([
        html.Div([dcc.Graph(figure=world_map)], style={'width': '100%'}),
        html.Div([
            html.Div([dcc.Graph(figure=region_pie)], style={'width': '33%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=gdp_sales)], style={'width': '33%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=countries_bar)], style={'width': '33%', 'display': 'inline-block'})
        ])
    ])

def create_predictive_tab():
    """Create the predictive analytics tab"""
    
    # Simple linear trend prediction
    sales_df['day_number'] = (sales_df['date'] - sales_df['date'].min()).dt.days
    z = np.polyfit(sales_df['day_number'], sales_df['sales'], 1)
    p = np.poly1d(z)
    
    # Generate future dates
    future_days = np.arange(sales_df['day_number'].max() + 1, sales_df['day_number'].max() + 31)
    future_dates = [sales_df['date'].max() + timedelta(days=int(d)) for d in future_days]
    future_sales = p(future_days)
    
    # Create prediction plot
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=sales_df['date'], y=sales_df['sales'], 
                                 name='Historical Sales', mode='lines', line=dict(color='blue')))
    pred_fig.add_trace(go.Scatter(x=future_dates, y=future_sales, 
                                 name='Predicted Sales', mode='lines', line=dict(color='red', dash='dash')))
    pred_fig.update_layout(title="Sales Prediction (Next 30 Days)", 
                          xaxis_title="Date", yaxis_title="Sales ($)", height=400)
    
    # Customer lifetime value prediction
    customer_df['predicted_ltv'] = customer_df['income'] * 0.15 + customer_df['satisfaction'] * 100
    ltv_hist = px.histogram(customer_df, x='predicted_ltv', nbins=20,
                           title="Predicted Customer Lifetime Value Distribution",
                           labels={'predicted_ltv': 'Predicted LTV ($)', 'count': 'Number of Customers'})
    ltv_hist.update_layout(height=400)
    
    # Seasonal decomposition (simplified)
    monthly_avg = sales_df.groupby('month')['sales'].mean()
    seasonal_fig = px.bar(x=monthly_avg.index, y=monthly_avg.values,
                         title="Seasonal Sales Pattern",
                         labels={'x': 'Month', 'y': 'Average Sales ($)'})
    seasonal_fig.update_layout(height=400)
    
    # Correlation heatmap
    numeric_cols = customer_df.select_dtypes(include=[np.number]).columns
    corr_matrix = customer_df[numeric_cols].corr()
    
    corr_heatmap = px.imshow(corr_matrix, 
                            title="Customer Data Correlation Matrix",
                            color_continuous_scale='RdBu_r')
    corr_heatmap.update_layout(height=400)
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=pred_fig)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=ltv_hist)], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=seasonal_fig)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=corr_heatmap)], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.H4("🔮 Predictive Insights"),
            html.Ul([
                html.Li(f"Predicted next month sales: ${p(sales_df['day_number'].max() + 30):,.0f}"),
                html.Li(f"Average predicted customer LTV: ${customer_df['predicted_ltv'].mean():,.0f}"),
                html.Li(f"Peak seasonal month: {monthly_avg.idxmax()}"),
                html.Li(f"Strongest correlation: Income vs Lifetime Value ({corr_matrix.loc['income', 'lifetime_value']:.2f})")
            ])
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ])

# =============================================================================
# RUN THE APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("🚀 Starting Interactive Data Visualization Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("🎨 Features:")
    print("   • Multi-tab interface with 6 different visualization categories")
    print("   • Interactive charts and real-time data updates")
    print("   • Responsive design with custom styling")
    print("   • Educational tooltips and insights")
    print("   • Predictive analytics and forecasting")
    print("\n💡 Use Ctrl+C to stop the server")
    
    app.run(debug=False, host='127.0.0.1', port=8050)
