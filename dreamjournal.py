"""
Dream Journal Visualizer - An Interactive Dream Analytics Dashboard
A sophisticated application for tracking, analyzing, and visualizing dreams
with ML-powered insights and surreal 3D visualizations.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter
import re

# ============================================================================
# MIDDLEWARE LAYER - Data Management & Processing
# ============================================================================

class DreamDataMiddleware:
    """Middleware for handling dream data operations and caching"""

    def __init__(self):
        self.dreams_db = []
        self.emotion_colors = {
            'happy': '#FFD700', 'sad': '#4169E1', 'fearful': '#8B0000',
            'anxious': '#FF6347', 'peaceful': '#90EE90', 'excited': '#FF69B4',
            'confused': '#9370DB', 'angry': '#DC143C', 'nostalgic': '#DDA0DD',
            'neutral': '#808080'
        }
        self.lucidity_levels = ['not_lucid', 'semi_lucid', 'fully_lucid']
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Generate rich sample dream data"""
        sample_dreams = [
            {
                'date': '2024-10-15', 'time': '03:30', 'title': 'Flying Over Ocean',
                'content': 'I was soaring above a vast ocean with bioluminescent waves. Dolphins were jumping alongside me, and I could breathe underwater. The sky was purple with three moons.',
                'emotions': ['peaceful', 'excited', 'happy'], 'intensity': 8,
                'lucidity': 'semi_lucid', 'characters': ['dolphins', 'myself'],
                'locations': ['ocean', 'sky'], 'symbols': ['water', 'flying', 'moon'],
                'colors': ['purple', 'blue', 'silver'], 'duration': 45
            },
            {
                'date': '2024-10-16', 'time': '04:15', 'title': 'Lost in Library',
                'content': 'Endless library with floating books. Each book contained memories of people I never met. The librarian was a shadowy figure who spoke in whispers. Time moved backwards.',
                'emotions': ['confused', 'curious', 'anxious'], 'intensity': 6,
                'lucidity': 'not_lucid', 'characters': ['librarian', 'shadow figures'],
                'locations': ['library', 'maze'], 'symbols': ['books', 'knowledge', 'time'],
                'colors': ['brown', 'gold', 'black'], 'duration': 30
            },
            {
                'date': '2024-10-18', 'time': '02:45', 'title': 'Childhood Home Transformed',
                'content': 'My childhood home but everything was giant-sized. I was tiny like an ant. Found my grandmother baking cookies in the kitchen, she had passed away years ago. She smiled and waved.',
                'emotions': ['nostalgic', 'sad', 'peaceful'], 'intensity': 9,
                'lucidity': 'not_lucid', 'characters': ['grandmother', 'family'],
                'locations': ['home', 'kitchen'], 'symbols': ['family', 'food', 'childhood'],
                'colors': ['warm', 'yellow', 'brown'], 'duration': 35
            },
            {
                'date': '2024-10-20', 'time': '05:00', 'title': 'Concert in the Void',
                'content': 'Playing piano in an empty void surrounded by stars. The music created colors and shapes that danced around me. Audience was made of pure light beings.',
                'emotions': ['excited', 'happy', 'peaceful'], 'intensity': 10,
                'lucidity': 'fully_lucid', 'characters': ['light beings', 'myself'],
                'locations': ['void', 'space'], 'symbols': ['music', 'stars', 'creation'],
                'colors': ['black', 'gold', 'rainbow'], 'duration': 60
            },
            {
                'date': '2024-10-22', 'time': '03:00', 'title': 'Chase Through City',
                'content': 'Being chased through a neon-lit city that kept shifting and changing. Buildings would appear and disappear. Finally hid in a cafe where everyone was frozen in time.',
                'emotions': ['fearful', 'anxious', 'confused'], 'intensity': 7,
                'lucidity': 'not_lucid', 'characters': ['pursuers', 'frozen people'],
                'locations': ['city', 'cafe'], 'symbols': ['chase', 'urban', 'time'],
                'colors': ['neon', 'blue', 'red'], 'duration': 25
            },
            {
                'date': '2024-10-24', 'time': '04:30', 'title': 'Garden of Memories',
                'content': 'Walking through a garden where each flower represented a memory. Some were wilting, others blooming. Found a fountain that showed reflections of past dreams.',
                'emotions': ['nostalgic', 'peaceful', 'happy'], 'intensity': 8,
                'lucidity': 'semi_lucid', 'characters': ['gardener', 'myself'],
                'locations': ['garden', 'fountain'], 'symbols': ['flowers', 'memory', 'water'],
                'colors': ['green', 'pink', 'gold'], 'duration': 50
            },
            {
                'date': '2024-10-25', 'time': '02:30', 'title': 'Mirror World',
                'content': 'Everything was reflected and reversed. Met my mirror self who had lived a completely different life. We exchanged stories through telepathy.',
                'emotions': ['confused', 'curious', 'excited'], 'intensity': 9,
                'lucidity': 'fully_lucid', 'characters': ['mirror self', 'reflections'],
                'locations': ['mirror world', 'reflection'], 'symbols': ['mirror', 'duality', 'self'],
                'colors': ['silver', 'blue', 'white'], 'duration': 40
            },
            {
                'date': '2024-10-26', 'time': '03:45', 'title': 'Underwater Temple',
                'content': 'Exploring ancient underwater ruins with glowing hieroglyphics. Fish swam through walls. Found a crystal that showed visions of the future.',
                'emotions': ['curious', 'excited', 'peaceful'], 'intensity': 8,
                'lucidity': 'semi_lucid', 'characters': ['ancient priests', 'sea creatures'],
                'locations': ['underwater', 'temple', 'ruins'], 'symbols': ['water', 'ancient', 'crystal'],
                'colors': ['blue', 'turquoise', 'gold'], 'duration': 55
            },
            {
                'date': '2024-10-27', 'time': '05:15', 'title': 'Storm of Emotions',
                'content': 'Standing in a field during a storm where each raindrop was an emotion. Could taste feelings. Lightning struck and created portals to other dreams.',
                'emotions': ['intense', 'confused', 'anxious'], 'intensity': 10,
                'lucidity': 'not_lucid', 'characters': ['storm entity', 'myself'],
                'locations': ['field', 'storm'], 'symbols': ['weather', 'emotion', 'portal'],
                'colors': ['grey', 'electric', 'dark'], 'duration': 30
            },
            {
                'date': '2024-10-28', 'time': '04:00', 'title': 'Quantum Market',
                'content': 'Bizarre marketplace where vendors sold impossible things: bottled time, crystallized thoughts, caged laughter. Paid with memories instead of money.',
                'emotions': ['curious', 'excited', 'confused'], 'intensity': 7,
                'lucidity': 'semi_lucid', 'characters': ['vendors', 'shoppers', 'myself'],
                'locations': ['market', 'bazaar'], 'symbols': ['trade', 'abstract', 'exchange'],
                'colors': ['vibrant', 'purple', 'gold'], 'duration': 45
            }
        ]

        # assign stable ids and coerce numeric fields
        for d in sample_dreams:
            d['id'] = hashlib.md5(f"{d['date']}{d['time']}{d.get('title','')}".encode()).hexdigest()[:8]
            # Ensure numeric types
            d['intensity'] = int(d.get('intensity', 0))
            d['duration'] = int(d.get('duration', 0))
            # ensure lists exist
            for k in ['emotions', 'characters', 'symbols']:
                d[k] = d.get(k, []) if isinstance(d.get(k, []), list) else [d.get(k)]
        self.dreams_db = sample_dreams

    def add_dream(self, dream_data):
        """Add new dream entry"""
        dream_data = dict(dream_data)  # copy
        dream_data.setdefault('date', datetime.now().strftime('%Y-%m-%d'))
        dream_data.setdefault('time', datetime.now().strftime('%H:%M'))
        dream_data['id'] = hashlib.md5(
            f"{dream_data['date']}{dream_data['time']}{dream_data.get('title','')}".encode()
        ).hexdigest()[:8]
        # coerce numeric
        try:
            dream_data['intensity'] = int(dream_data.get('intensity', 0))
        except Exception:
            dream_data['intensity'] = 0
        try:
            dream_data['duration'] = int(dream_data.get('duration', 0))
        except Exception:
            dream_data['duration'] = 0
        self.dreams_db.append(dream_data)
        return True

    def get_all_dreams(self):
        """Retrieve all dreams as DataFrame with correct types"""
        if not self.dreams_db:
            return pd.DataFrame()
        df = pd.DataFrame(self.dreams_db)
        # Ensure columns exist and types
        if 'intensity' in df.columns:
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce').fillna(0).astype(int)
        else:
            df['intensity'] = 0
        if 'duration' in df.columns:
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0).astype(int)
        else:
            df['duration'] = 0
        df['date'] = df['date'].astype(str)
        df['time'] = df['time'].astype(str)
        df['lucidity'] = df.get('lucidity', pd.Series(['not_lucid'] * len(df))).fillna('not_lucid')
        return df

    def get_dream_by_id(self, dream_id):
        """Get specific dream"""
        for dream in self.dreams_db:
            if dream.get('id') == dream_id:
                return dream
        return None

    def compute_emotion_stats(self):
        """Calculate emotion statistics"""
        if not self.dreams_db:
            return {}
        emotion_counts = Counter()
        for dream in self.dreams_db:
            emotion_counts.update(dream.get('emotions', []))
        return dict(emotion_counts)

    def analyze_text_patterns(self):
        """NLP analysis of dream content"""
        if not self.dreams_db:
            return None

        texts = [d.get('content', '') for d in self.dreams_db]
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        return {
            'tfidf_matrix': tfidf_matrix,
            'feature_names': feature_names,
            'vectorizer': vectorizer
        }

    def cluster_dreams(self, n_clusters=3):
        """Cluster dreams using K-Means"""
        analysis = self.analyze_text_patterns()
        if analysis is None:
            return None
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(analysis['tfidf_matrix'])
        return clusters

    def reduce_dimensions(self, method='tsne'):
        """Dimensionality reduction for visualization"""
        analysis = self.analyze_text_patterns()
        if analysis is None:
            return None

        tfidf_dense = analysis['tfidf_matrix'].toarray()
        n_samples = tfidf_dense.shape[0]

        # If too few samples for TSNE, fallback to PCA or return zeros
        if n_samples < 2:
            return np.zeros((n_samples, 3))

        if method == 'pca' or n_samples < 4:
            reducer = PCA(n_components=min(3, n_samples), random_state=42)
            coords = reducer.fit_transform(tfidf_dense)
            # If result is lower-dim, pad to 3 cols
            if coords.shape[1] < 3:
                coords = np.pad(coords, ((0,0),(0,3-coords.shape[1])), mode='constant')
            return coords
        elif method == 'tsne':
            # TSNE perplexity must be < n_samples, choose safe value
            perplexity = min(30, max(2, n_samples - 1))
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity, init='random')
        else:
            reducer = MDS(n_components=3, random_state=42)
        coords = reducer.fit_transform(tfidf_dense)
        # ensure shape (n,3)
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0,0),(0,3-coords.shape[1])), mode='constant')
        return coords

    def build_character_network(self):
        """Build network graph of dream characters"""
        G = nx.Graph()
        for dream in self.dreams_db:
            chars = dream.get('characters', []) or []
            # normalize char names to strings
            chars = [str(c).strip() for c in chars if c]
            for char in chars:
                if not G.has_node(char):
                    G.add_node(char, count=1)
                else:
                    G.nodes[char]['count'] += 1
            for i, char1 in enumerate(chars):
                for char2 in chars[i+1:]:
                    if G.has_edge(char1, char2):
                        G[char1][char2]['weight'] += 1
                    else:
                        G.add_edge(char1, char2, weight=1)
        return G

    def analyze_temporal_patterns(self):
        """Analyze patterns over time"""
        df = self.get_all_dreams()
        if df.empty:
            return None
        # Combine date and time safely
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        df = df.sort_values('datetime')
        df['intensity_ma'] = df['intensity'].rolling(window=3, min_periods=1).mean()
        df['duration_ma'] = df['duration'].rolling(window=3, min_periods=1).mean()
        return df

# ============================================================================
# VISUALIZATION GENERATORS
# ============================================================================

class DreamVisualizer:
    """Generate all visualization figures"""

    def __init__(self, middleware):
        self.mw = middleware

    def create_dreamscape_3d(self):
        """Create 3D dreamscape visualization using dimensionality reduction"""
        coords = self.mw.reduce_dimensions(method='tsne')
        if coords is None or coords.size == 0:
            return go.Figure()
        df = self.mw.get_all_dreams()
        # Ensure coords align with df rows
        if coords.shape[0] != len(df):
            # try PCA fallback
            coords = self.mw.reduce_dimensions(method='pca')

        # Create color mapping
        colors = []
        for emotions in df['emotions']:
            primary_emotion = emotions[0] if isinstance(emotions, (list, tuple)) and emotions else 'neutral'
            colors.append(self.mw.emotion_colors.get(primary_emotion, '#808080'))

        # Size based on intensity
        sizes = (df['intensity'].astype(int).values * 5).tolist()
        # customdata for hover (date, intensity)
        customdata = np.stack([df['date'].astype(str).values, df['intensity'].astype(str).values], axis=1)

        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            text=df['title'].values,
            textposition='top center',
            textfont=dict(size=8, color='white'),
            customdata=customdata,
            hovertemplate='<b>%{text}</b><br>Date: %{customdata[0]}<br>Intensity: %{customdata[1]}<extra></extra>',
            name='Dreams'
        )])

        # Add connecting lines for similar dreams
        analysis = self.mw.analyze_text_patterns()
        if analysis is not None:
            try:
                similarity_matrix = cosine_similarity(analysis['tfidf_matrix'])
                threshold = 0.3
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        if similarity_matrix[i, j] > threshold:
                            fig.add_trace(go.Scatter3d(
                                x=[coords[i, 0], coords[j, 0]],
                                y=[coords[i, 1], coords[j, 1]],
                                z=[coords[i, 2], coords[j, 2]],
                                mode='lines',
                                line=dict(color='rgba(255,255,255,0.08)', width=1),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
            except Exception:
                # similarity might fail for edge cases; ignore gracefully
                pass

        fig.update_layout(
            template='plotly_dark',
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''),
                bgcolor='rgba(0,0,0,0.9)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            title=dict(
                text='🌌 Dreamscape: 3D Dream Space',
                font=dict(size=20, color='#00E3AE')
            )
        )
        return fig

    def create_emotion_wheel(self):
        """Create circular emotion distribution chart"""
        emotion_stats = self.mw.compute_emotion_stats()
        if not emotion_stats:
            return go.Figure()
        emotions = list(emotion_stats.keys())
        counts = list(emotion_stats.values())
        colors = [self.mw.emotion_colors.get(e, '#808080') for e in emotions]
        fig = go.Figure(data=[go.Pie(
            labels=emotions,
            values=counts,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
        )])
        fig.update_layout(
            template='plotly_dark',
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            title=dict(text='😊 Emotion Wheel', font=dict(size=18, color='#FFD700')),
            showlegend=True,
            legend=dict(orientation='v', yanchor='middle', y=0.5)
        )
        return fig

    def create_character_network(self):
        """Create network graph of dream characters"""
        G = self.mw.build_character_network()
        if len(G.nodes()) == 0:
            return go.Figure()

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]].get('weight', 1)
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=max(1, weight * 1.5), color='rgba(255,255,255,0.3)'),
                hoverinfo='none',
                showlegend=False
            ))

        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node].get('count', 1) * 8 + 8)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color='#7F7EFF',
                line=dict(color='white', width=2)
            ),
            text=node_text,
            textposition='top center',
            textfont=dict(size=10, color='white'),
            hovertemplate='<b>%{text}</b><br>Appearances: %{marker.size}<extra></extra>',
            showlegend=False
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            template='plotly_dark',
            height=450,
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, t=60, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=dict(text='👥 Character Network', font=dict(size=18, color='#7F7EFF'))
        )
        return fig

    def create_temporal_heatmap(self):
        """Create calendar heatmap / timeline of dream intensity"""
        df = self.mw.analyze_temporal_patterns()
        if df is None or df.empty:
            return go.Figure()
        # Use scatter timeline with markers sized by intensity
        fig = go.Figure(data=go.Scatter(
            x=df['datetime'],
            y=df['intensity'],
            mode='markers+lines',
            marker=dict(
                size=(df['intensity'] * 5).tolist(),
                color=df['intensity'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Intensity')
            ),
            line=dict(color='rgba(127,126,255,0.3)', width=2),
            text=df['title'],
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Intensity: %{y}<extra></extra>'
        ))
        fig.update_layout(
            template='plotly_dark',
            height=350,
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis_title='Date',
            yaxis_title='Intensity',
            title=dict(text='📅 Dream Intensity Timeline', font=dict(size=18, color='#00E3AE'))
        )
        return fig

    def create_lucidity_gauge(self):
        """Create gauge chart for lucidity levels"""
        df = self.mw.get_all_dreams()
        if df.empty:
            return go.Figure()
        lucidity_counts = df['lucidity'].value_counts()
        total = len(df)
        fully_lucid_pct = (lucidity_counts.get('fully_lucid', 0) / total) * 100
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=fully_lucid_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Lucidity Rate (%)', 'font': {'size': 20, 'color': 'white'}},
            delta={'reference': 20, 'increasing': {'color': '#00E3AE'}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white'},
                'bar': {'color': '#7F7EFF'},
                'bgcolor': 'rgba(0,0,0,0.3)',
                'borderwidth': 2,
                'bordercolor': 'white',
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(127,126,255,0.2)'},
                    {'range': [33, 66], 'color': 'rgba(127,126,255,0.4)'},
                    {'range': [66, 100], 'color': 'rgba(127,126,255,0.6)'}
                ],
                'threshold': {
                    'line': {'color': '#00E3AE', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(template='plotly_dark', height=300, margin=dict(l=20, r=20, t=60, b=20))
        return fig

    def create_word_cloud_chart(self):
        """Create word frequency bar chart (TF-IDF based)"""
        analysis = self.mw.analyze_text_patterns()
        if analysis is None:
            return go.Figure()
        feature_names = analysis['feature_names']
        tfidf_matrix = analysis['tfidf_matrix']
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        if len(scores) == 0:
            return go.Figure()
        top_n = min(20, len(scores))
        top_indices = scores.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = scores[top_indices]
        fig = go.Figure(data=[go.Bar(
            x=top_scores,
            y=top_words,
            orientation='h',
            marker=dict(
                color=top_scores,
                colorscale='Plasma',
                showscale=False
            ),
            hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
        )])
        fig.update_layout(
            template='plotly_dark',
            height=500,
            margin=dict(l=100, r=20, t=60, b=40),
            xaxis_title='TF-IDF Score',
            yaxis_title='',
            title=dict(text='💬 Top Dream Keywords', font=dict(size=18, color='#FFD700'))
        )
        return fig

    def create_symbol_sunburst(self):
        """Create sunburst chart of dream symbols"""
        df = self.mw.get_all_dreams()
        if df.empty:
            return go.Figure()
        all_symbols = []
        for symbols in df['symbols']:
            if isinstance(symbols, (list, tuple)):
                all_symbols.extend(symbols)
            elif symbols:
                all_symbols.append(symbols)
        symbol_counts = Counter(all_symbols)
        if not symbol_counts:
            return go.Figure()
        labels = ['All Symbols'] + list(symbol_counts.keys())
        parents = [''] + ['All Symbols'] * len(symbol_counts)
        values = [sum(symbol_counts.values())] + list(symbol_counts.values())
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(colorscale='Viridis', line=dict(color='white', width=2)),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        ))
        fig.update_layout(
            template='plotly_dark',
            height=450,
            margin=dict(l=0, r=0, t=60, b=0),
            title=dict(text='🔮 Dream Symbols Hierarchy', font=dict(size=18, color='#9370DB'))
        )
        return fig

# ============================================================================
# DASH APPLICATION
# ============================================================================

# Initialize middleware and visualizer
middleware = DreamDataMiddleware()
visualizer = DreamVisualizer(middleware)

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "💭 Dream Journal Visualizer"

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1('💭 Dream Journal Visualizer',
                   style={'margin': 0, 'fontSize': '3em', 'fontWeight': 'bold'}),
            html.P('Explore the landscape of your subconscious mind',
                  style={'fontSize': '1.3em', 'opacity': 0.9, 'marginTop': '10px'})
        ], style={'textAlign': 'center'})
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
        'padding': '40px 20px',
        'borderRadius': '12px',
        'margin': '20px',
        'boxShadow': '0 8px 32px rgba(0,0,0,0.3)'
    }),

    # Stats Cards
    html.Div([
        html.Div([
            html.H3('📊', style={'fontSize': '2.5em', 'margin': 0}),
            html.H2(id='total-dreams', style={'margin': '10px 0'}),
            html.P('Total Dreams', style={'opacity': 0.8})
        ], style={
            'background': 'rgba(102,126,234,0.2)',
            'padding': '20px',
            'borderRadius': '10px',
            'textAlign': 'center',
            'flex': 1,
            'minWidth': '150px'
        }),

        html.Div([
            html.H3('⭐', style={'fontSize': '2.5em', 'margin': 0}),
            html.H2(id='avg-intensity', style={'margin': '10px 0'}),
            html.P('Avg Intensity', style={'opacity': 0.8})
        ], style={
            'background': 'rgba(255,215,0,0.2)',
            'padding': '20px',
            'borderRadius': '10px',
            'textAlign': 'center',
            'flex': 1,
            'minWidth': '150px'
        }),

        html.Div([
            html.H3('🌟', style={'fontSize': '2.5em', 'margin': 0}),
            html.H2(id='lucid-count', style={'margin': '10px 0'}),
            html.P('Lucid Dreams', style={'opacity': 0.8})
        ], style={
            'background': 'rgba(127,126,255,0.2)',
            'padding': '20px',
            'borderRadius': '10px',
            'textAlign': 'center',
            'flex': 1,
            'minWidth': '150px'
        }),

        html.Div([
            html.H3('⏱️', style={'fontSize': '2.5em', 'margin': 0}),
            html.H2(id='avg-duration', style={'margin': '10px 0'}),
            html.P('Avg Duration (min)', style={'opacity': 0.8})
        ], style={
            'background': 'rgba(0,227,174,0.2)',
            'padding': '20px',
            'borderRadius': '10px',
            'textAlign': 'center',
            'flex': 1,
            'minWidth': '150px'
        })
    ], style={'display': 'flex', 'gap': '16px', 'justifyContent': 'space-between', 'margin': '20px'}),

    # Main visualization grid
    html.Div([
        html.Div([
            dcc.Graph(id='dreamscape-3d'),
            dcc.Graph(id='emotion-wheel')
        ], style={'flex': '1', 'minWidth': '420px', 'padding': '10px'}),

        html.Div([
            dcc.Graph(id='word-cloud'),
            dcc.Graph(id='symbol-sunburst')
        ], style={'flex': '1', 'minWidth': '420px', 'padding': '10px'})
    ], style={'display': 'flex', 'gap': '16px', 'margin': '10px 20px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='character-network')
        ], style={'flex': '1', 'minWidth': '420px', 'padding': '10px'}),

        html.Div([
            dcc.Graph(id='temporal-plot'),
            dcc.Graph(id='lucidity-gauge')
        ], style={'flex': '1', 'minWidth': '420px', 'padding': '10px'})
    ], style={'display': 'flex', 'gap': '16px', 'margin': '10px 20px 40px 20px'}),

    dcc.Interval(id='update-interval', interval=60*1000, n_intervals=0)
])

@app.callback(
    Output('total-dreams', 'children'),
    Output('avg-intensity', 'children'),
    Output('lucid-count', 'children'),
    Output('avg-duration', 'children'),
    Output('dreamscape-3d', 'figure'),
    Output('emotion-wheel', 'figure'),
    Output('word-cloud', 'figure'),
    Output('symbol-sunburst', 'figure'),
    Output('character-network', 'figure'),
    Output('temporal-plot', 'figure'),
    Output('lucidity-gauge', 'figure'),
    Input('update-interval', 'n_intervals')
)
def update_dashboard(n_intervals):
    """Update all dashboard elements periodically."""
    df = middleware.get_all_dreams()
    total = len(df)
    avg_int = round(df['intensity'].mean(), 2) if total > 0 else 0
    lucid_count = int(df[df['lucidity'] == 'fully_lucid'].shape[0]) if total > 0 else 0
    avg_dur = round(df['duration'].mean(), 1) if total > 0 else 0

    # Generate figures using visualizer
    dreamscape_fig = visualizer.create_dreamscape_3d()
    emotion_fig = visualizer.create_emotion_wheel()
    word_fig = visualizer.create_word_cloud_chart()
    symbol_fig = visualizer.create_symbol_sunburst()
    char_net_fig = visualizer.create_character_network()
    temporal_fig = visualizer.create_temporal_heatmap()
    lucidity_fig = visualizer.create_lucidity_gauge()

    return (
        str(total),
        str(avg_int),
        str(lucid_count),
        str(avg_dur),
        dreamscape_fig,
        emotion_fig,
        word_fig,
        symbol_fig,
        char_net_fig,
        temporal_fig,
        lucidity_fig
    )

if __name__ == '__main__':
    # When run directly, start the Dash development server
    app.run_server(debug=True, port=8050)
