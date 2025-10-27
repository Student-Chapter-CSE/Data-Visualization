
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge, Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import os
import sys
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_palette("viridis")

def create_sample_data():
    np.random.seed(42)
    
    time_points = pd.date_range('2023-01-01', periods=365, freq='D')
    crypto_prices = 50000 + np.cumsum(np.random.normal(0, 1000, 365))
    social_sentiment = np.random.normal(0.5, 0.2, 365)
    trading_volume = np.random.lognormal(10, 1, 365)
    
    crypto_data = pd.DataFrame({
        'Date': time_points,
        'Price': crypto_prices,
        'Sentiment': social_sentiment,
        'Volume': trading_volume,
        'Volatility': np.abs(np.random.normal(0, 0.05, 365))
    })
    
    cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Dubai', 'Singapore', 'Toronto']
    city_data = pd.DataFrame({
        'City': cities,
        'Population': np.random.randint(500000, 15000000, 8),
        'GDP': np.random.uniform(50, 2000, 8),
        'Tech_Score': np.random.uniform(0, 100, 8),
        'Innovation_Index': np.random.uniform(0, 10, 8),
        'Latitude': [40.7128, 51.5074, 35.6762, 48.8566, -33.8688, 25.2048, 1.3521, 43.6532],
        'Longitude': [-74.0060, -0.1278, 139.6503, 2.3522, 151.2093, 55.2708, 103.8198, -79.3832]
    })
    
    n_points = 200
    cluster_data = pd.DataFrame({
        'X': np.random.normal(0, 1, n_points),
        'Y': np.random.normal(0, 1, n_points),
        'Z': np.random.normal(0, 1, n_points),
        'Cluster': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education'], n_points),
        'Size': np.random.uniform(10, 100, n_points),
        'Growth': np.random.uniform(-0.2, 0.5, n_points)
    })
    
    network_nodes = 20
    network_data = {
        'nodes': [{'id': i, 'label': f'Node_{i}', 'value': np.random.uniform(1, 10)} for i in range(network_nodes)],
        'edges': [(i, j) for i in range(network_nodes) for j in range(i+1, network_nodes) if np.random.random() < 0.3]
    }
    
    return crypto_data, city_data, cluster_data, network_data

def setup_git_branch(branch_name="quantum-data-viz"):
    """Setup git branch for the quantum data visualization project."""
    try:
        print(f"🔧 Setting up git branch: {branch_name}")
        
        # Check if git is initialized
        if not os.path.exists('.git'):
            print("   📁 Initializing git repository...")
            subprocess.run(['git', 'init'], check=True)
        
        # Create and checkout new branch
        print(f"   🌿 Creating branch: {branch_name}")
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        
        # Add all files
        print("   📦 Adding files to staging...")
        subprocess.run(['git', 'add', '.'], check=True)
        
        # Create initial commit
        commit_message = f"🚀 Initial commit: {branch_name} - Next-Gen Data Intelligence Suite"
        print(f"   💾 Creating commit: {commit_message}")
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        
        print(f"   ✅ Git branch '{branch_name}' created successfully!")
        print(f"   📋 Branch status:")
        subprocess.run(['git', 'status'], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Git error: {e}")
        return False
    except FileNotFoundError:
        print("   ❌ Git not found. Please install Git first.")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def create_crypto_radar_chart(df):
    fig = go.Figure()
    
    for i in range(0, len(df), 30):
        subset = df.iloc[i:i+30]
        avg_sentiment = subset['Sentiment'].mean()
        avg_volatility = subset['Volatility'].mean()
        avg_volume = np.log(subset['Volume']).mean()
        price_change = (subset['Price'].iloc[-1] - subset['Price'].iloc[0]) / subset['Price'].iloc[0]
        
        fig.add_trace(go.Scatterpolar(
            r=[avg_sentiment, avg_volatility*100, avg_volume, abs(price_change)*100, 1-avg_volatility],
            theta=['Sentiment', 'Volatility', 'Volume', 'Price Change', 'Stability'],
            fill='toself',
            name=f'Period {i//30 + 1}',
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Crypto Market Radar Analysis",
        font=dict(size=12, color='white'),
        paper_bgcolor='black',
        plot_bgcolor='black'
    )
    return fig

def create_3d_city_bubble_chart(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=df['Longitude'],
        y=df['Latitude'],
        z=df['Tech_Score'],
        mode='markers',
        marker=dict(
            size=df['Population']/100000,
            color=df['Innovation_Index'],
            colorscale='Viridis',
            opacity=0.8,
            line=dict(width=2, color='white')
        ),
        text=df['City'],
        hovertemplate='<b>%{text}</b><br>' +
                     'Population: %{marker.size:,.0f}<br>' +
                     'Tech Score: %{z:.1f}<br>' +
                     'Innovation: %{marker.color:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Global Tech Cities 3D Bubble Map",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude", 
            zaxis_title="Tech Score",
            bgcolor='black'
        ),
        font=dict(color='white'),
        paper_bgcolor='black'
    )
    return fig

def create_ai_cluster_analysis(df):
    fig = plt.figure(figsize=(15, 10))
    
    scaler = StandardScaler()
    features = scaler.fit_transform(df[['X', 'Y', 'Z']])
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(df['X'], df['Y'], df['Z'], c=clusters, cmap='viridis', s=df['Size'], alpha=0.7)
    ax1.set_title('AI-Powered 3D Clustering', color='white', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature X', color='white')
    ax1.set_ylabel('Feature Y', color='white')
    ax1.set_zlabel('Feature Z', color='white')
    
    ax2 = fig.add_subplot(222)
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    scatter2 = ax2.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='viridis', s=df['Size'], alpha=0.7)
    ax2.set_title('PCA Dimensionality Reduction', color='white', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', color='white')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', color='white')
    
    ax3 = fig.add_subplot(223)
    growth_by_cluster = df.groupby(clusters)['Growth'].mean()
    bars = ax3.bar(range(len(growth_by_cluster)), growth_by_cluster, color=plt.cm.viridis(np.linspace(0, 1, len(growth_by_cluster))))
    ax3.set_title('Average Growth by Cluster', color='white', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Cluster', color='white')
    ax3.set_ylabel('Growth Rate', color='white')
    ax3.set_xticks(range(len(growth_by_cluster)))
    
    ax4 = fig.add_subplot(224)
    cluster_sizes = np.bincount(clusters)
    wedges, texts, autotexts = ax4.pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(len(cluster_sizes))], 
                                       autopct='%1.1f%%', colors=plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes))))
    ax4.set_title('Cluster Distribution', color='white', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_network_visualization(network_data):
    G = nx.Graph()
    
    for node in network_data['nodes']:
        G.add_node(node['id'], label=node['label'], value=node['value'])
    
    for edge in network_data['edges']:
        G.add_edge(edge[0], edge[1])
    
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    node_sizes = [network_data['nodes'][i]['value'] * 100 for i in range(len(network_data['nodes']))]
    node_colors = [network_data['nodes'][i]['value'] for i in range(len(network_data['nodes']))]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          cmap='plasma', alpha=0.8, ax=ax)
    
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='white', width=1, ax=ax)
    
    nx.draw_networkx_labels(G, pos, {i: f'Node {i}' for i in range(len(network_data['nodes']))}, 
                           font_size=8, font_color='white', ax=ax)
    
    ax.set_title('Interactive Network Graph', color='white', fontsize=16, fontweight='bold')
    ax.set_facecolor('black')
    
    plt.tight_layout()
    return fig

def create_animated_crypto_candlestick(df):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Price'] * (1 - df['Volatility']),
        high=df['Price'] * (1 + df['Volatility']),
        low=df['Price'] * (1 - df['Volatility']),
        close=df['Price'],
        name="Crypto Price",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines',
        name='Price Trend',
        line=dict(color='white', width=2),
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Real-time Crypto Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        font=dict(color='white'),
        paper_bgcolor='black',
        plot_bgcolor='black',
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date'
        ),
        yaxis=dict(fixedrange=False)
    )
    
    return fig

def create_quantum_style_visualization(df):
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(221)
    ax1.scatter(df['X'], df['Y'], c=df['Z'], s=df['Size']*10, 
               cmap='plasma', alpha=0.7, edgecolors='white', linewidth=0.5)
    ax1.set_title('Quantum Scatter Field', color='white', fontsize=14, fontweight='bold')
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.3, color='cyan')
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(df['X'], df['Y'], df['Z'], c=df['Growth'], s=df['Size']*5, 
               cmap='viridis', alpha=0.8)
    ax2.set_title('3D Quantum Field', color='white', fontsize=14, fontweight='bold')
    ax2.set_facecolor('black')
    
    ax3 = fig.add_subplot(223)
    hexbin = ax3.hexbin(df['X'], df['Y'], C=df['Z'], gridsize=20, cmap='plasma', alpha=0.8)
    ax3.set_title('Hexagonal Density Map', color='white', fontsize=14, fontweight='bold')
    ax3.set_facecolor('black')
    plt.colorbar(hexbin, ax=ax3, label='Z Value')
    
    ax4 = fig.add_subplot(224)
    contour = ax4.tricontourf(df['X'], df['Y'], df['Z'], levels=20, cmap='inferno', alpha=0.8)
    ax4.set_title('Contour Field Visualization', color='white', fontsize=14, fontweight='bold')
    ax4.set_facecolor('black')
    plt.colorbar(contour, ax=ax4, label='Field Strength')
    
    plt.tight_layout()
    return fig

def create_holistic_dashboard(crypto_df, city_df, cluster_df, network_data):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Crypto Price Evolution', 'Global Tech Cities', 
                       'AI Cluster Analysis', 'Network Connections',
                       'Market Volatility', 'Innovation Heatmap'),
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    fig.add_trace(
        go.Scatter(x=crypto_df['Date'], y=crypto_df['Price'], 
                  name='Crypto Price', line=dict(color='#00ff88', width=3),
                  hovertemplate='Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter3d(x=city_df['Longitude'], y=city_df['Latitude'], z=city_df['Tech_Score'],
                    mode='markers', name='Cities',
                    marker=dict(size=city_df['Population']/200000, color=city_df['Innovation_Index'],
                               colorscale='Viridis', opacity=0.8),
                    text=city_df['City'], hovertemplate='<b>%{text}</b><br>Tech Score: %{z}<extra></extra>'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=cluster_df['X'], y=cluster_df['Y'], 
                  mode='markers', name='Clusters',
                  marker=dict(size=cluster_df['Size'], color=cluster_df['Growth'],
                             colorscale='Plasma', opacity=0.7),
                  hovertemplate='Growth: %{marker.color:.2f}<extra></extra>'),
        row=2, col=1
    )
    
    G = nx.Graph()
    for node in network_data['nodes']:
        G.add_node(node['id'])
    for edge in network_data['edges']:
        G.add_edge(edge[0], edge[1])
    pos = nx.spring_layout(G)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode='lines', name='Network',
                  line=dict(color='white', width=1), opacity=0.5),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=crypto_df['Date'], y=crypto_df['Volatility'], 
                  mode='lines+markers', name='Volatility',
                  line=dict(color='#ff4444', width=2),
                  hovertemplate='Volatility: %{y:.3f}<extra></extra>'),
        row=3, col=1
    )
    
    heatmap_data = np.random.rand(10, 10)
    fig.add_trace(
        go.Heatmap(z=heatmap_data, colorscale='Hot', name='Innovation'),
        row=3, col=2
    )
    
    fig.update_layout(
        title_text="🚀 Next-Gen Data Intelligence Dashboard",
        title_x=0.5,
        showlegend=False,
        height=1200,
        font=dict(color='white'),
        paper_bgcolor='black',
        plot_bgcolor='black'
    )
    
    return fig

def main():
    print("🚀 Initializing Next-Gen Data Intelligence Suite...")
    
    # Setup git branch
    print("\n🔧 Git Configuration:")
    git_success = setup_git_branch("quantum-data-viz")
    
    if git_success:
        print("   ✅ Git setup completed successfully!")
    else:
        print("   ⚠️ Git setup failed, continuing with visualizations...")
    
    print("\n📊 Generating advanced datasets...")
    crypto_df, city_df, cluster_df, network_data = create_sample_data()
    
    print("📊 Generated advanced datasets:")
    print(f"Crypto data: {crypto_df.shape[0]} days of market data")
    print(f"Global cities: {city_df.shape[0]} tech hubs analyzed")
    print(f"AI clusters: {cluster_df.shape[0]} data points for ML analysis")
    print(f"Network nodes: {len(network_data['nodes'])} interconnected entities")
    
    print("\n🎨 Creating cutting-edge visualizations...")
    
    print("   🌟 Crypto Market Radar Analysis...")
    radar_fig = create_crypto_radar_chart(crypto_df)
    radar_fig.show()
    
    print("   🌍 3D Global Tech Cities Map...")
    city_fig = create_3d_city_bubble_chart(city_df)
    city_fig.show()
    
    print("   🤖 AI-Powered Cluster Analysis...")
    cluster_fig = create_ai_cluster_analysis(cluster_df)
    cluster_fig.show()
    
    print("   🕸️ Interactive Network Visualization...")
    network_fig = create_network_visualization(network_data)
    network_fig.show()
    
    print("   📈 Real-time Crypto Candlestick...")
    candlestick_fig = create_animated_crypto_candlestick(crypto_df)
    candlestick_fig.show()
    
    print("   ⚛️ Quantum-Style Field Visualization...")
    quantum_fig = create_quantum_style_visualization(cluster_df)
    quantum_fig.show()
    
    print("\n🎯 Creating holistic intelligence dashboard...")
    dashboard = create_holistic_dashboard(crypto_df, city_df, cluster_df, network_data)
    dashboard.show()
    
    print("\n✅ Next-Gen Data Intelligence Suite Complete!")
    print("\n🚀 Revolutionary Features:")
    print("   • AI-powered clustering with PCA dimensionality reduction")
    print("   • 3D interactive global tech city mapping")
    print("   • Real-time crypto market radar analysis")
    print("   • Network graph theory visualization")
    print("   • Quantum-inspired field visualizations")
    print("   • Multi-dimensional data intelligence dashboard")
    print("   • Dark theme with neon aesthetics")
    print("   • Machine learning integration")
    print("   • Advanced statistical analysis")
    print("   • Interactive hover effects and animations")

if __name__ == "__main__":
    main()
