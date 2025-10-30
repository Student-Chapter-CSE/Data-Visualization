
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime
import colorsys
import random

class DreamConstellationMap:
    def __init__(self):
        # Sample dream data (you can replace this with your own data)
        self.dreams = [
            {
                "theme": "Flying",
                "emotions": ["freedom", "joy", "excitement"],
                "symbols": ["wings", "clouds", "birds"],
                "intensity": 8,
                "date": "2024-10-15",
                "color_tone": "bright",
                "dreamscape": "sky"
            },
            {
                "theme": "Ocean Depths",
                "emotions": ["mystery", "peace", "wonder"],
                "symbols": ["fish", "coral", "waves"],
                "intensity": 7,
                "date": "2024-10-16",
                "color_tone": "deep",
                "dreamscape": "underwater"
            },
            {
                "theme": "Ancient Temple",
                "emotions": ["awe", "curiosity", "reverence"],
                "symbols": ["statues", "inscriptions", "torch"],
                "intensity": 9,
                "date": "2024-10-17",
                "color_tone": "warm",
                "dreamscape": "indoor"
            }
        ]
        
    def _create_star_coordinates(self, n_stars, spread=1):
        """Generate star coordinates in a spiral galaxy pattern"""
        theta = np.random.uniform(0, 2*np.pi, n_stars)
        radius = np.random.normal(loc=0.5, scale=0.2, size=n_stars) * spread
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.random.normal(0, 0.1, n_stars)
        return x, y, z
    
    def _generate_star_colors(self, n_stars):
        """Generate realistic star colors based on temperature"""
        # Simulate star temperatures (2000K to 12000K)
        temperatures = np.random.uniform(2000, 12000, n_stars)
        colors = []
        
        for temp in temperatures:
            # Approximate star color based on temperature
            if temp < 3500:
                # Red stars
                hue = 0.0
            elif temp < 5000:
                # Orange/Yellow stars
                hue = 0.08
            elif temp < 6000:
                # Yellow stars
                hue = 0.16
            elif temp < 7500:
                # White stars
                hue = 0.6
            else:
                # Blue stars
                hue = 0.7
                
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, 0.3, 1.0)
            colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
            
        return colors
    
    def _create_nebula_effect(self, n_points=1000):
        """Create a colorful nebula effect in the background"""
        x = np.random.normal(0, 1, n_points)
        y = np.random.normal(0, 1, n_points)
        z = np.random.normal(0, 0.1, n_points)
        
        # Create different colored nebula regions
        colors = []
        for _ in range(n_points):
            hue = random.choice([0.7, 0.1, 0.9])  # Blue, Red, Purple
            sat = random.uniform(0.5, 1.0)
            val = random.uniform(0.1, 0.3)
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},0.1)')
        
        return x, y, z, colors

    def create_constellation_map(self):
        """Create the main constellation map visualization"""
        # Create figure
        fig = go.Figure()
        
        # Add nebula background
        neb_x, neb_y, neb_z, neb_colors = self._create_nebula_effect(2000)
        fig.add_trace(go.Scatter3d(
            x=neb_x, y=neb_y, z=neb_z,
            mode='markers',
            marker=dict(
                size=2,
                color=neb_colors,
                opacity=0.1
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Create star field
        n_stars = 500
        star_x, star_y, star_z = self._create_star_coordinates(n_stars)
        star_colors = self._generate_star_colors(n_stars)
        
        # Add background stars
        fig.add_trace(go.Scatter3d(
            x=star_x, y=star_y, z=star_z,
            mode='markers',
            marker=dict(
                size=2,
                color=star_colors,
                opacity=0.8
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Create dream theme nodes
        themes = [dream['theme'] for dream in self.dreams]
        theme_x, theme_y, theme_z = self._create_star_coordinates(len(themes), spread=0.5)
        
        # Add dream theme "constellations"
        fig.add_trace(go.Scatter3d(
            x=theme_x, y=theme_y, z=theme_z,
            mode='markers+text',
            marker=dict(
                size=10,
                color=['#FFD700', '#7EB0D5', '#B793F5'],
                symbol='star',
                opacity=0.8
            ),
            text=themes,
            textposition='top center',
            hovertemplate=(
                "<b>Theme:</b> %{text}<br>" +
                "<b>Emotions:</b> %{customdata[0]}<br>" +
                "<b>Symbols:</b> %{customdata[1]}<br>" +
                "<b>Intensity:</b> %{customdata[2]}"
            ),
            customdata=[
                [
                    ', '.join(dream['emotions']),
                    ', '.join(dream['symbols']),
                    dream['intensity']
                ] for dream in self.dreams
            ],
            name='Dream Themes'
        ))
        
        # Connect related themes with constellation lines
        for i in range(len(themes)-1):
            fig.add_trace(go.Scatter3d(
                x=[theme_x[i], theme_x[i+1]],
                y=[theme_y[i], theme_y[i+1]],
                z=[theme_z[i], theme_z[i+1]],
                mode='lines',
                line=dict(
                    color='rgba(255, 255, 255, 0.3)',
                    width=2
                ),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Update layout for cosmic theme
        fig.update_layout(
            template='plotly_dark',
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, visible=False),
                yaxis=dict(showgrid=False, showticklabels=False, visible=False),
                zaxis=dict(showgrid=False, showticklabels=False, visible=False),
                bgcolor='rgba(0,0,0,0.95)',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
            ),
            title=dict(
                text='🌌 Dream Constellation Map',
                font=dict(size=24, color='#E3E3E3'),
                y=0.95
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig

if __name__ == "__main__":
    # Create visualization
    constellation_map = DreamConstellationMap()
    fig = constellation_map.create_constellation_map()
    
    # Show the interactive visualization
    fig.show()