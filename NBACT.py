import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import requests
from io import BytesIO
from PIL import Image
import base64
import time
from functools import lru_cache
import json
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'NBA2025.xlsx'
HEADSHOT_BASE_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/"
LOGO_BASE_URL = "https://cdn.nba.com/logos/nba/{}/primary/L/logo.svg"
CACHE_TTL = 3600  # 1 hour cache
MAX_PLAYERS = 500  # Limit for performance reasons

# Load team metadata
TEAM_METADATA = {
    'ATL': {'id': 1610612737, 'name': 'Atlanta Hawks', 'colors': ['#E03A3E', '#C1D32F']},
    'BOS': {'id': 1610612738, 'name': 'Boston Celtics', 'colors': ['#007A33', '#BA9653']},
    'BKN': {'id': 1610612751, 'name': 'Brooklyn Nets', 'colors': ['#000000', '#FFFFFF']},
    'CHA': {'id': 1610612766, 'name': 'Charlotte Hornets', 'colors': ['#1D1160', '#00788C']},
    'CHI': {'id': 1610612741, 'name': 'Chicago Bulls', 'colors': ['#CE1141', '#000000']},
    'CLE': {'id': 1610612739, 'name': 'Cleveland Cavaliers', 'colors': ['#860038', '#041E42']},
    'DAL': {'id': 1610612742, 'name': 'Dallas Mavericks', 'colors': ['#00538C', '#002B5E']},
    'DEN': {'id': 1610612743, 'name': 'Denver Nuggets', 'colors': ['#0E2240', '#FEC524']},
    'DET': {'id': 1610612765, 'name': 'Detroit Pistons', 'colors': ['#C8102E', '#1D42BA']},
    'GSW': {'id': 1610612744, 'name': 'Golden State Warriors', 'colors': ['#1D428A', '#FFC72C']},
    'HOU': {'id': 1610612745, 'name': 'Houston Rockets', 'colors': ['#CE1141', '#000000']},
    'IND': {'id': 1610612754, 'name': 'Indiana Pacers', 'colors': ['#002D62', '#FDBB30']},
    'LAC': {'id': 1610612746, 'name': 'LA Clippers', 'colors': ['#C8102E', '#1D428A']},
    'LAL': {'id': 1610612747, 'name': 'Los Angeles Lakers', 'colors': ['#552583', '#FDB927']},
    'MEM': {'id': 1610612763, 'name': 'Memphis Grizzlies', 'colors': ['#5D76A9', '#12173F']},
    'MIA': {'id': 1610612748, 'name': 'Miami Heat', 'colors': ['#98002E', '#F9A01B']},
    'MIL': {'id': 1610612749, 'name': 'Milwaukee Bucks', 'colors': ['#00471B', '#EEE1C6']},
    'MIN': {'id': 1610612750, 'name': 'Minnesota Timberwolves', 'colors': ['#0C2340', '#236192']},
    'NOP': {'id': 1610612740, 'name': 'New Orleans Pelicans', 'colors': ['#0C2340', '#85714D']},
    'NYK': {'id': 1610612752, 'name': 'New York Knicks', 'colors': ['#006BB6', '#F58426']},
    'OKC': {'id': 1610612760, 'name': 'Oklahoma City Thunder', 'colors': ['#007AC1', '#EF3B24']},
    'ORL': {'id': 1610612753, 'name': 'Orlando Magic', 'colors': ['#0077C0', '#C4CED4']},
    'PHI': {'id': 1610612755, 'name': 'Philadelphia 76ers', 'colors': ['#006BB6', '#ED174C']},
    'PHX': {'id': 1610612756, 'name': 'Phoenix Suns', 'colors': ['#1D1160', '#E56020']},
    'POR': {'id': 1610612757, 'name': 'Portland Trail Blazers', 'colors': ['#E03A3E', '#000000']},
    'SAC': {'id': 1610612758, 'name': 'Sacramento Kings', 'colors': ['#5A2D81', '#63727A']},
    'SAS': {'id': 1610612759, 'name': 'San Antonio Spurs', 'colors': ['#C4CED4', '#000000']},
    'TOR': {'id': 1610612761, 'name': 'Toronto Raptors', 'colors': ['#CE1141', '#000000']},
    'UTA': {'id': 1610612762, 'name': 'Utah Jazz', 'colors': ['#002B5C', '#00471B']},
    'WAS': {'id': 1610612764, 'name': 'Washington Wizards', 'colors': ['#002B5C', '#E31837']}
}

# Improved data loading with better error handling and caching
@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading NBA data...")
def load_data():
    """Enhanced data loading with comprehensive preprocessing and validation"""
    try:
        # Check if cached processed data exists
        cache_file = 'nba_data_cache.feather'
        if os.path.exists(cache_file):
            try:
                df = pd.read_feather(cache_file)
                if not df.empty:
                    return df
            except:
                pass
        
        # Load data with multiple sheets handling
        xls = pd.ExcelFile(DATA_FILE)
        
        # Try common sheet names if 'NBA2025' doesn't exist
        sheet_name = 'NBA2025'
        if sheet_name not in xls.sheet_names:
            for name in ['Data', 'Players', 'Stats', 'Sheet1']:
                if name in xls.sheet_names:
                    sheet_name = name
                    break
        
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('%', 'pct').str.lower()
        
        # Column name mapping with case insensitivity
        column_mapping = {
            'player': 'player',
            'name': 'player',
            'player_name': 'player',
            'tm': 'team',
            'team': 'team',
            'pos': 'position',
            'position': 'position',
            'age': 'age',
            'pts': 'points',
            'trb': 'rebounds',
            'ast': 'assists',
            'stl': 'steals',
            'blk': 'blocks',
            'fg': 'fg_made',
            'fga': 'fg_attempted',
            'fg_pct': 'fg_pct',
            '3p': '3p_made',
            '3pa': '3p_attempted',
            '3p_pct': '3p_pct',
            '2p': '2p_made',
            '2pa': '2p_attempted',
            '2p_pct': '2p_pct',
            'efg_pct': 'efg_pct',
            'ft': 'ft_made',
            'fta': 'ft_attempted',
            'ft_pct': 'ft_pct',
            'orb': 'offensive_rebounds',
            'drb': 'defensive_rebounds',
            'tov': 'turnovers',
            'pf': 'personal_fouls',
            'g': 'games',
            'gs': 'games_started',
            'mp': 'minutes_played',
            'per': 'per',
            'ts_pct': 'ts_pct',
            'usg_pct': 'usg_pct'
        }
        
        # Apply column name standardization
        df = df.rename(columns=lambda x: column_mapping.get(x.lower(), x))
        
        if 'player' not in df.columns:
            st.error("Could not find player name column in data")
            return pd.DataFrame()
            
        # Ensure required columns exist with defaults
        required_columns = {
            'team': 'UNK',
            'position': 'UNK',
            'age': 25,
            'points': 0,
            'rebounds': 0,
            'assists': 0,
            'steals': 0,
            'blocks': 0,
            'minutes_played': 0,
            'fg_made': 0,
            'fg_attempted': 0,
            '3p_made': 0,
            '3p_attempted': 0,
            'ft_made': 0,
            'ft_attempted': 0,
            'offensive_rebounds': 0,
            'defensive_rebounds': 0,
            'turnovers': 0,
            'personal_fouls': 0,
            'games': 0,
            'games_started': 0
        }
        
        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default
        
        # Numeric columns processing with better type handling
        numeric_cols = ['age', 'games', 'games_started', 'minutes_played', 'fg_made', 'fg_attempted', 
                       '3p_made', '3p_attempted', '2p_made', '2p_attempted', 'ft_made', 
                       'ft_attempted', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 
                       'assists', 'steals', 'blocks', 'turnovers', 'personal_fouls', 'points']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Handle outliers using robust scaling
                q1 = df[col].quantile(0.05)
                q3 = df[col].quantile(0.95)
                df[col] = np.where(df[col] < q1, q1, df[col])
                df[col] = np.where(df[col] > q3, q3, df[col])
        
        # Calculate advanced metrics with safeguards
        df['minutes_played'] = df['minutes_played'].replace(0, 0.001)  # Avoid division by zero
        
        if 'per' not in df.columns:
            df['per'] = (df['points'] + df['rebounds'] + df['assists'] + 
                         df['steals'] + df['blocks']) / df['minutes_played'] * 36
        
        if 'ts_pct' not in df.columns:
            df['ts_pct'] = df['points'] / (2 * (df['fg_attempted'] + 0.44 * df['ft_attempted'] + 0.001))
        
        if 'usg_pct' not in df.columns:
            df['usg_pct'] = (df['fg_attempted'] + 0.44 * df['ft_attempted'] + df['turnovers']) * 100 / df['minutes_played']
        
        df['ast_to'] = df['assists'] / (df['turnovers'] + 0.001)
        df['stl_blk'] = (df['steals'] + df['blocks']) / df['minutes_played'] * 36
        
        # Calculate eFG% if not already in data
        if 'efg_pct' not in df.columns and 'fg_made' in df.columns and 'fg_attempted' in df.columns and '3p_made' in df.columns:
            df['efg_pct'] = (df['fg_made'] + 0.5 * df['3p_made']) / df['fg_attempted']
        elif 'efg_pct' not in df.columns:
            df['efg_pct'] = 0
        
        # Position classification with more categories
        if 'position' in df.columns:
            df['position_group'] = df['position'].str[:1].replace({
                'G': 'Guard',
                'F': 'Forward',
                'C': 'Center'
            }).fillna('Other')
            
            # Add more detailed position classification
            df['position_detail'] = df['position'].map({
                'PG': 'Point Guard',
                'SG': 'Shooting Guard',
                'SF': 'Small Forward',
                'PF': 'Power Forward',
                'C': 'Center',
                'G': 'Guard',
                'F': 'Forward'
            }).fillna('Other')
        else:
            df['position_group'] = 'Unknown'
            df['position_detail'] = 'Unknown'
        
        # Enhanced player value metric
        df['player_value'] = (
            df['points'] * 0.25 + 
            df['rebounds'] * 0.15 + 
            df['assists'] * 0.15 + 
            df['steals'] * 0.1 + 
            df['blocks'] * 0.1 + 
            df['ts_pct'] * 0.1 +
            df['per'] * 0.1 +
            df['efg_pct'] * 0.05
        )
        
        # Create consistent player ID from name
        df['player_id'] = (
            df['player']
            .str.lower()
            .str.replace(' ', '-')
            .str.replace("'", "")
            .str.replace(".", "")
            .str.replace("ć", "c")
            .str.replace("š", "s")
            .str.replace("đ", "d")
            .str.replace("č", "c")
            .str.replace("ž", "z")
        )
        
        # Add per-game stats
        df['minutes_per_game'] = df['minutes_played'] / df['games']
        df['points_per_game'] = df['points'] / df['games']
        df['rebounds_per_game'] = df['rebounds'] / df['games']
        df['assists_per_game'] = df['assists'] / df['games']
        df['steals_per_game'] = df['steals'] / df['games']
        df['blocks_per_game'] = df['blocks'] / df['games']
        
        # Add per-minute stats
        df['points_per_36'] = df['points'] / df['minutes_played'] * 36
        df['rebounds_per_36'] = df['rebounds'] / df['minutes_played'] * 36
        df['assists_per_36'] = df['assists'] / df['minutes_played'] * 36
        
        # Add shooting metrics
        df['fga_per_game'] = df['fg_attempted'] / df['games']
        df['3pa_per_game'] = df['3p_attempted'] / df['games']
        df['fta_per_game'] = df['ft_attempted'] / df['games']
        
        # Save to cache
        try:
            df.to_feather(cache_file)
        except:
            pass
            
        return df.fillna(0).replace([np.inf, -np.inf], 0)
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()
        return pd.DataFrame()

# Improved image fetching with caching and retries
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_image(url, timeout=5, retries=2):
    """Fetch image with retries and timeout handling"""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            time.sleep(1)  # Delay between retries
        except Exception:
            time.sleep(1)
    return None

def get_player_headshot(player_id):
    """Get player headshot image with multiple fallback options"""
    # Try standard NBA CDN first
    standard_url = f"{HEADSHOT_BASE_URL}{player_id}.png"
    img = fetch_image(standard_url)
    if img:
        return img
    
    # Try common variations if standard fails
    variations = [
        player_id,
        player_id.split('-')[0] + '-' + player_id.split('-')[1][0],
        player_id.replace("-", ""),
        player_id.replace("-", "").lower(),
        player_id.replace("-", " ").lower().replace(" ", "-"),
        player_id.replace("-", " ").title().replace(" ", "-")
    ]
    
    for variation in variations:
        variation_url = f"{HEADSHOT_BASE_URL}{variation}.png"
        img = fetch_image(variation_url)
        if img:
            return img
    
    return None

def get_team_logo(team_abbrev):
    """Get team logo with comprehensive team mapping"""
    if not team_abbrev or pd.isna(team_abbrev) or team_abbrev not in TEAM_METADATA:
        return None
    
    team_id = TEAM_METADATA[team_abbrev]['id']
    
    # Try SVG first
    svg_url = LOGO_BASE_URL.format(team_id)
    img = fetch_image(svg_url)
    if img:
        return img
    
    # Fallback to PNG
    png_url = f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.png"
    return fetch_image(png_url)

def get_team_colors(team_abbrev):
    """Get team's primary and secondary colors"""
    if team_abbrev in TEAM_METADATA:
        return TEAM_METADATA[team_abbrev]['colors']
    return ['#4caf50', '#f44336']  # Default colors

def image_to_base64(image):
    """Convert PIL image to base64 with format detection"""
    if not image:
        return ""
    
    buffered = BytesIO()
    # Try to preserve original format if possible
    format = image.format if hasattr(image, 'format') else "PNG"
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def create_comparison_df(player1, player2, df):
    """Create comprehensive comparison DataFrame with categorized metrics"""
    try:
        p1 = df[df['player'] == player1].iloc[0] if player1 in df['player'].values else None
        p2 = df[df['player'] == player2].iloc[0] if player2 in df['player'].values else None
        
        if p1 is None or p2 is None:
            return pd.DataFrame()
        
        # Organized metrics by category with weights for importance
        metrics = {
            'Scoring': {
                'metrics': ['points', 'points_per_game', 'points_per_36', 'fg_pct', '3p_pct', 'ft_pct', 'ts_pct', 'efg_pct'],
                'weight': 0.25
            },
            'Rebounding': {
                'metrics': ['offensive_rebounds', 'defensive_rebounds', 'rebounds', 'rebounds_per_game', 'rebounds_per_36'],
                'weight': 0.15
            },
            'Playmaking': {
                'metrics': ['assists', 'assists_per_game', 'assists_per_36', 'ast_to', 'turnovers'],
                'weight': 0.15
            },
            'Defense': {
                'metrics': ['steals', 'blocks', 'steals_per_game', 'blocks_per_game', 'stl_blk', 'personal_fouls'],
                'weight': 0.15
            },
            'Efficiency': {
                'metrics': ['per', 'ts_pct', 'usg_pct', 'efg_pct', 'ast_to'],
                'weight': 0.15
            },
            'Volume': {
                'metrics': ['minutes_played', 'minutes_per_game', 'fg_attempted', 'fga_per_game', '3p_attempted', '3pa_per_game', 'ft_attempted', 'fta_per_game'],
                'weight': 0.1
            },
            'Durability': {
                'metrics': ['games', 'games_started'],
                'weight': 0.05
            }
        }
        
        comparison_data = []
        for category, data in metrics.items():
            for metric in data['metrics']:
                if metric in p1 and metric in p2:
                    p1_val = p1[metric]
                    p2_val = p2[metric]
                    diff = p1_val - p2_val
                    percent_diff = (diff / (abs(p2_val) + 0.001)) * 100  # Avoid division by zero
                    
                    # Normalize difference based on metric scale
                    max_val = max(abs(p1_val), abs(p2_val))
                    normalized_diff = diff / (max_val + 0.001) if max_val != 0 else 0
                    
                    comparison_data.append({
                        'category': category,
                        'metric': metric,
                        'player1': p1_val,
                        'player2': p2_val,
                        'difference': diff,
                        'percent_difference': percent_diff,
                        'normalized_difference': normalized_diff,
                        'weight': data['weight']
                    })
        
        return pd.DataFrame(comparison_data)
    except Exception as e:
        st.error(f"Comparison error: {str(e)}")
        return pd.DataFrame()

def calculate_similarity(player1, player2, df):
    """Calculate multiple similarity metrics with position adjustment"""
    try:
        # Features to consider with weights
        feature_weights = {
            'points_per_game': 0.12,
            'rebounds_per_game': 0.1,
            'assists_per_game': 0.1,
            'steals_per_game': 0.05,
            'blocks_per_game': 0.05,
            'fg_pct': 0.08,
            '3p_pct': 0.08,
            'ft_pct': 0.07,
            'per': 0.1,
            'ts_pct': 0.07,
            'usg_pct': 0.05,
            'efg_pct': 0.05,
            'ast_to': 0.05,
            'stl_blk': 0.03
        }
        
        # Only use features that exist in the dataframe
        features = [f for f in feature_weights.keys() if f in df.columns]
        weights = np.array([feature_weights[f] for f in features])
        
        if not features:
            return {'cosine': 0, 'euclidean': 0, 'combined': 0}
        
        # Prepare data
        numeric_df = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Get player indices
        p1_idx = df[df['player'] == player1].index[0]
        p2_idx = df[df['player'] == player2].index[0]
        
        # Calculate weighted similarity metrics
        cosine_sim = cosine_similarity(
            (scaled_data[p1_idx] * weights).reshape(1, -1),
            (scaled_data[p2_idx] * weights).reshape(1, -1)
        )[0][0]
        
        euclidean_dist = euclidean_distances(
            (scaled_data[p1_idx] * weights).reshape(1, -1),
            (scaled_data[p2_idx] * weights).reshape(1, -1)
        )[0][0]
        
        # Normalize Euclidean to 0-1 scale (inverse)
        max_dist = np.sqrt(np.sum(weights**2) * (4**2))  # Assuming 4 SD max difference
        normalized_euclidean = 1 - (euclidean_dist / max_dist)
        
        # Position similarity adjustment
        pos_similarity = 0
        if 'position_detail' in df.columns:
            pos1 = df.at[p1_idx, 'position_detail']
            pos2 = df.at[p2_idx, 'position_detail']
            
            # Position similarity matrix
            position_groups = {
                'Point Guard': 'Guard',
                'Shooting Guard': 'Guard',
                'Small Forward': 'Forward',
                'Power Forward': 'Forward',
                'Center': 'Center'
            }
            
            group1 = position_groups.get(pos1, 'Other')
            group2 = position_groups.get(pos2, 'Other')
            
            if pos1 == pos2:
                pos_similarity = 0.15  # Same exact position
            elif group1 == group2:
                pos_similarity = 0.1  # Same general position group
            else:
                pos_similarity = 0  # Different positions
        
        combined_score = (cosine_sim * 0.4 + normalized_euclidean * 0.45 + pos_similarity)
        
        return {
            'cosine': cosine_sim,
            'euclidean': normalized_euclidean,
            'combined': min(1.0, max(0.0, combined_score))  # Ensure between 0-1
        }
    except Exception as e:
        st.error(f"Similarity calculation error: {str(e)}")
        return {'cosine': 0, 'euclidean': 0, 'combined': 0}

def create_radar_chart(player1, player2, comparison_df, player1_color='#4caf50', player2_color='#f44336'):
    """Create an interactive radar chart comparing player metrics by category"""
    try:
        # Group metrics by category and calculate weighted averages
        categories = comparison_df['category'].unique()
        
        # Prepare data for radar chart
        p1_values = []
        p2_values = []
        category_labels = []
        
        for category in categories:
            cat_df = comparison_df[comparison_df['category'] == category]
            # Calculate weighted average for each category
            p1_avg = np.average(cat_df['player1'], weights=cat_df['weight'])
            p2_avg = np.average(cat_df['player2'], weights=cat_df['weight'])
            p1_values.append(p1_avg)
            p2_values.append(p2_avg)
            category_labels.append(category)
        
        # Normalize values to 0-1 scale for better visualization
        max_val = max(max(p1_values), max(p2_values)) + 0.1  # Add small buffer
        p1_values = [v / max_val for v in p1_values]
        p2_values = [v / max_val for v in p2_values]
        
        # Create radar chart with custom styling
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=p1_values + [p1_values[0]],  # Close the shape
            theta=category_labels + [category_labels[0]],
            fill='toself',
            name=player1,
            line=dict(color=player1_color, width=2),
            fillcolor=f'rgba({int(player1_color[1:3], 16)}, {int(player1_color[3:5], 16)}, {int(player1_color[5:7], 16)}, 0.3)',
            hoverinfo='text',
            hovertext=[f"{player1}<br>{cat}: {val:.2f}" for cat, val in zip(category_labels, p1_values)]
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=p2_values + [p2_values[0]],  # Close the shape
            theta=category_labels + [category_labels[0]],
            fill='toself',
            name=player2,
            line=dict(color=player2_color, width=2),
            fillcolor=f'rgba({int(player2_color[1:3], 16)}, {int(player2_color[3:5], 16)}, {int(player2_color[5:7], 16)}, 0.3)',
            hoverinfo='text',
            hovertext=[f"{player2}<br>{cat}: {val:.2f}" for cat, val in zip(category_labels, p2_values)]
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickvals=[0, 0.5, 1],
                    ticktext=['0', '0.5', '1'],
                    tickfont=dict(size=10),
                    gridcolor='rgba(255, 255, 255, 0.2)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.5)'
                ),
                bgcolor='rgba(0, 0, 0, 0)'
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            hoverlabel=dict(
                bgcolor="rgba(30, 33, 48, 0.9)",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Radar chart error: {str(e)}")
        return go.Figure()

def create_player_similarity_plot(player1, player2, df, top_n=10):
    """Create a scatter plot showing similar players in 2D space"""
    try:
        # Features to use for similarity
        features = [
            'points_per_game', 'rebounds_per_game', 'assists_per_game',
            'fg_pct', '3p_pct', 'per', 'usg_pct', 'efg_pct'
        ]
        
        # Filter to only features we have
        features = [f for f in features if f in df.columns]
        
        if not features:
            return None
            
        # Prepare data
        numeric_df = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Reduce to 2D with PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(reduced_data, columns=['x', 'y'])
        plot_df['player'] = df['player']
        plot_df['position'] = df['position_detail']
        plot_df['team'] = df['team']
        plot_df['value'] = df['player_value']
        
        # Get indices of our target players
        p1_idx = df[df['player'] == player1].index[0]
        p2_idx = df[df['player'] == player2].index[0]
        
        # Get most similar players to each
        similarity_matrix = cosine_similarity(scaled_data)
        p1_similar = similarity_matrix[p1_idx].argsort()[::-1][1:top_n+1]
        p2_similar = similarity_matrix[p2_idx].argsort()[::-1][1:top_n+1]
        
        # Create figure
        fig = go.Figure()
        
        # Add all players as faint background
        fig.add_trace(go.Scatter(
            x=plot_df['x'],
            y=plot_df['y'],
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(200, 200, 200, 0.2)',
                line=dict(width=0)
            ),
            text=plot_df['player'] + '<br>' + plot_df['position'] + '<br>' + plot_df['team'],
            hoverinfo='text',
            name='All Players'
        ))
        
        # Add similar players to player 1
        fig.add_trace(go.Scatter(
            x=plot_df.iloc[p1_similar]['x'],
            y=plot_df.iloc[p1_similar]['y'],
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(76, 175, 80, 0.7)',
                line=dict(width=1, color='white')
            ),
            text=plot_df.iloc[p1_similar]['player'] + '<br>' + plot_df.iloc[p1_similar]['position'] + '<br>' + plot_df.iloc[p1_similar]['team'],
            hoverinfo='text',
            name=f'Similar to {player1}'
        ))
        
        # Add similar players to player 2
        fig.add_trace(go.Scatter(
            x=plot_df.iloc[p2_similar]['x'],
            y=plot_df.iloc[p2_similar]['y'],
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(244, 67, 54, 0.7)',
                line=dict(width=1, color='white')
            ),
            text=plot_df.iloc[p2_similar]['player'] + '<br>' + plot_df.iloc[p2_similar]['position'] + '<br>' + plot_df.iloc[p2_similar]['team'],
            hoverinfo='text',
            name=f'Similar to {player2}'
        ))
        
        # Add target players
        fig.add_trace(go.Scatter(
            x=[plot_df.iloc[p1_idx]['x'], plot_df.iloc[p2_idx]['x']],
            y=[plot_df.iloc[p1_idx]['y'], plot_df.iloc[p2_idx]['y']],
            mode='markers',
            marker=dict(
                size=12,
                color=['rgba(76, 175, 80, 1)', 'rgba(244, 67, 54, 1)'],
                line=dict(width=2, color='white')
            ),
            text=[player1, player2],
            hoverinfo='text',
            name='Selected Players'
        ))
        
        # Add annotations for target players
        fig.add_annotation(
            x=plot_df.iloc[p1_idx]['x'],
            y=plot_df.iloc[p1_idx]['y'],
            text=player1,
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(color='white')
        )
        
        fig.add_annotation(
            x=plot_df.iloc[p2_idx]['x'],
            y=plot_df.iloc[p2_idx]['y'],
            text=player2,
            showarrow=True,
            arrowhead=1,
            ax=-20,
            ay=30,
            font=dict(color='white')
        )
        
        # Update layout
        fig.update_layout(
            title=f'Player Similarity Landscape: {player1} vs {player2}',
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=600,
            hoverlabel=dict(
                bgcolor="rgba(30, 33, 48, 0.9)",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Similarity plot error: {str(e)}")
        return None

def generate_comprehensive_report(player1, player2, comparison_df, similarity_scores, df):
    """Generate detailed analytical report with actionable insights"""
    try:
        # Get player data with error handling
        try:
            p1 = df[df['player'] == player1].iloc[0]
            p2 = df[df['player'] == player2].iloc[0]
        except IndexError:
            st.error("One or both players not found in dataset")
            return ""
        
        # Get team colors for styling
        team1_colors = get_team_colors(p1.get('team', ''))
        team2_colors = get_team_colors(p2.get('team', ''))
        player1_color = team1_colors[0] if team1_colors else '#4caf50'
        player2_color = team2_colors[0] if team2_colors else '#f44336'
        
        # Calculate overall scores
        p1_score = np.average(comparison_df['player1'], weights=comparison_df['weight'])
        p2_score = np.average(comparison_df['player2'], weights=comparison_df['weight'])
        overall_diff = p1_score - p2_score
        better_player = player1 if overall_diff > 0 else player2
        
        # Position analysis
        pos1 = p1.get('position_detail', 'N/A')
        pos2 = p2.get('position_detail', 'N/A')
        pos_comparison = "Same position" if pos1 == pos2 else f"Different positions ({pos1} vs {pos2})"
        
        # Similarity interpretation
        similarity_score = similarity_scores['combined']
        if similarity_score > 0.8:
            similarity_text = "Very Similar Play Styles"
        elif similarity_score > 0.6:
            similarity_text = "Similar Play Styles"
        elif similarity_score > 0.4:
            similarity_text = "Somewhat Similar"
        else:
            similarity_text = "Different Play Styles"
        
        # Create report with enhanced styling and sections
        report = f"""
        <style>
            .report-container {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #e0e0e0;
                max-width: 1000px;
                margin: 0 auto;
            }}
            .report-header {{
                text-align: center;
                padding: 20px 0;
                border-bottom: 2px solid #1e88e5;
                margin-bottom: 30px;
            }}
            .report-title {{
                color: #1e88e5;
                font-size: 28px;
                margin-bottom: 5px;
            }}
            .report-subtitle {{
                color: #bbbbbb;
                font-size: 18px;
                margin-top: 0;
            }}
            .section {{
                margin-bottom: 30px;
                background-color: #1e2130;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .section-title {{
                color: #1e88e5;
                font-size: 22px;
                margin-top: 0;
                margin-bottom: 15px;
                border-bottom: 1px solid #2d303e;
                padding-bottom: 10px;
            }}
            .metric-card {{
                background-color: #2d303e;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 15px;
            }}
            .metric-title {{
                font-weight: bold;
                margin-bottom: 8px;
                color: #bbbbbb;
            }}
            .metric-value {{
                font-size: 18px;
                color: #ffffff;
            }}
            .player1-color {{
                color: {player1_color};
                font-weight: bold;
            }}
            .player2-color {{
                color: {player2_color};
                font-weight: bold;
            }}
            .insight-box {{
                background-color: #2d303e;
                border-left: 4px solid #1e88e5;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 6px 6px 0;
            }}
            .comparison-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }}
            .stat-row {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }}
            .stat-name {{
                font-weight: bold;
            }}
            .stat-value {{
                font-family: 'Courier New', monospace;
            }}
            .recommendation {{
                background-color: #2d303e;
                border-radius: 6px;
                padding: 15px;
                margin: 10px 0;
            }}
            .recommendation-title {{
                color: #1e88e5;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            .similarity-badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
                background-color: {player1_color};
                color: white;
                margin-right: 5px;
            }}
            @media (max-width: 768px) {{
                .comparison-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        
        <div class="report-container">
            <div class="report-header">
                <h1 class="report-title">🏀 NBA Player Comparison Report 2025</h1>
                <h2 class="report-subtitle">
                    <span class="player1-color">{player1}</span> vs <span class="player2-color">{player2}</span>
                </h2>
            </div>
            
            <div class="section">
                <h3 class="section-title">📊 Executive Summary</h3>
                
                <div class="comparison-grid">
                    <div class="metric-card">
                        <div class="metric-title">Overall Advantage</div>
                        <div class="metric-value">
                            <span class="{'player1-color' if overall_diff > 0 else 'player2-color'}">{better_player}</span>
                            <br>({abs(overall_diff):.1f}% difference)
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Similarity Score</div>
                        <div class="metric-value">
                            <span class="similarity-badge">{similarity_score:.2f}/1.0</span>
                            <small>{similarity_text}</small>
                        </div>
                    </div>
                </div>
                
                <div class="insight-box">
                    <p><strong>Position Comparison:</strong> {pos_comparison}</p>
                    <p><strong>Player Value Score:</strong> <span class="player1-color">{p1['player_value']:.1f}</span> vs <span class="player2-color">{p2['player_value']:.1f}</span></p>
                    <p><strong>Key Takeaway:</strong> {better_player} provides more overall value based on comprehensive metrics analysis.</p>
                </div>
            </div>
            
            <div class="section">
                <h3 class="section-title">📈 Statistical Breakdown</h3>
                
                <div class="comparison-grid">
                    <div>
                        <h4>Per Game Stats</h4>
                        <div class="stat-row">
                            <span class="stat-name">Points:</span>
                            <span class="stat-value {'player1-color' if p1['points_per_game'] > p2['points_per_game'] else 'player2-color'}">
                                {p1['points_per_game']:.1f} vs {p2['points_per_game']:.1f}
                            </span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">Rebounds:</span>
                            <span class="stat-value {'player1-color' if p1['rebounds_per_game'] > p2['rebounds_per_game'] else 'player2-color'}">
                                {p1['rebounds_per_game']:.1f} vs {p2['rebounds_per_game']:.1f}
                            </span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">Assists:</span>
                            <span class="stat-value {'player1-color' if p1['assists_per_game'] > p2['assists_per_game'] else 'player2-color'}">
                                {p1['assists_per_game']:.1f} vs {p2['assists_per_game']:.1f}
                            </span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">Minutes:</span>
                            <span class="stat-value {'player1-color' if p1['minutes_per_game'] > p2['minutes_per_game'] else 'player2-color'}">
                                {p1['minutes_per_game']:.1f} vs {p2['minutes_per_game']:.1f}
                            </span>
                        </div>
                    </div>
                    
                    <div>
                        <h4>Efficiency Metrics</h4>
                        <div class="stat-row">
                            <span class="stat-name">PER:</span>
                            <span class="stat-value {'player1-color' if p1['per'] > p2['per'] else 'player2-color'}">
                                {p1['per']:.1f} vs {p2['per']:.1f}
                            </span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">TS%:</span>
                            <span class="stat-value {'player1-color' if p1['ts_pct'] > p2['ts_pct'] else 'player2-color'}">
                                {p1['ts_pct']:.3f} vs {p2['ts_pct']:.3f}
                            </span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">eFG%:</span>
                            <span class="stat-value {'player1-color' if p1['efg_pct'] > p2['efg_pct'] else 'player2-color'}">
                                {p1['efg_pct']:.3f} vs {p2['efg_pct']:.3f}
                            </span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">USG%:</span>
                            <span class="stat-value {'player1-color' if p1['usg_pct'] > p2['usg_pct'] else 'player2-color'}">
                                {p1['usg_pct']:.1f} vs {p2['usg_pct']:.1f}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3 class="section-title">🔍 Category Analysis</h3>
        """
        
        # Add category analysis
        categories = comparison_df['category'].unique()
        for category in categories:
            cat_df = comparison_df[comparison_df['category'] == category]
            p1_cat = np.average(cat_df['player1'], weights=cat_df['weight'])
            p2_cat = np.average(cat_df['player2'], weights=cat_df['weight'])
            diff = p1_cat - p2_cat
            percent_diff = (diff / (abs(p2_cat) + 0.001)) * 100
            
            report += f"""
                <div class="metric-card">
                    <div class="metric-title">{category}</div>
                    <div class="metric-value">
                        <span class="{'player1-color' if diff > 0 else 'player2-color'}">
                            {player1 if diff > 0 else player2} leads by {abs(diff):.1f} ({abs(percent_diff):.1f}%)
                        </span>
                    </div>
                    <div style="margin-top: 10px;">
                        <small>Key metrics: {', '.join(cat_df['metric'].unique())}</small>
                    </div>
                </div>
            """
        
        # Add key insights
        report += """
            </div>
            
            <div class="section">
                <h3 class="section-title">🏆 Scouting Recommendations</h3>
        """
        
        # Generate dynamic recommendations
        recommendations = []
        
        # Scoring recommendation
        if 'Scoring' in categories:
            scoring_diff = np.average(
                comparison_df[comparison_df['category'] == 'Scoring']['player1'],
                weights=comparison_df[comparison_df['category'] == 'Scoring']['weight']
            ) - np.average(
                comparison_df[comparison_df['category'] == 'Scoring']['player2'],
                weights=comparison_df[comparison_df['category'] == 'Scoring']['weight']
            )
            
            if abs(scoring_diff) > 0.1:  # Significant difference
                better_scorer = player1 if scoring_diff > 0 else player2
                recommendations.append({
                    'title': "Scoring Needs",
                    'content': f"{better_scorer} is the better scorer with a {abs(scoring_diff):.1f} advantage in scoring metrics"
                })
        
        # Defense recommendation
        if 'Defense' in categories:
            defense_diff = np.average(
                comparison_df[comparison_df['category'] == 'Defense']['player1'],
                weights=comparison_df[comparison_df['category'] == 'Defense']['weight']
            ) - np.average(
                comparison_df[comparison_df['category'] == 'Defense']['player2'],
                weights=comparison_df[comparison_df['category'] == 'Defense']['weight']
            )
            
            if abs(defense_diff) > 0.1:  # Significant difference
                better_defender = player1 if defense_diff > 0 else player2
                recommendations.append({
                    'title': "Defensive Needs",
                    'content': f"{better_defender} provides better defensive impact with a {abs(defense_diff):.1f} advantage in defensive metrics"
                })
        
        # Playmaking recommendation
        if 'Playmaking' in categories:
            playmaking_diff = np.average(
                comparison_df[comparison_df['category'] == 'Playmaking']['player1'],
                weights=comparison_df[comparison_df['category'] == 'Playmaking']['weight']
            ) - np.average(
                comparison_df[comparison_df['category'] == 'Playmaking']['player2'],
                weights=comparison_df[comparison_df['category'] == 'Playmaking']['weight']
            )
            
            if abs(playmaking_diff) > 0.1:
                better_playmaker = player1 if playmaking_diff > 0 else player2
                recommendations.append({
                    'title': "Playmaking Needs",
                    'content': f"{better_playmaker} is the better playmaker with a {abs(playmaking_diff):.1f} advantage in playmaking metrics"
                })
        
        # Position-specific recommendations
        if pos1 == pos2:
            if 'Guard' in pos1:
                recommendations.append({
                    'title': "Guard Skills",
                    'content': "Consider who fits your system better - scoring guard or playmaker"
                })
            elif 'Forward' in pos1:
                recommendations.append({
                    'title': "Forward Skills",
                    'content': "Consider versatility - inside/outside game, switchability on defense"
                })
            elif 'Center' in pos1:
                recommendations.append({
                    'title': "Center Skills",
                    'content': "Consider modern vs traditional center skills - spacing vs rim protection"
                })
        
        # Add all recommendations
        for rec in recommendations:
            report += f"""
                <div class="recommendation">
                    <div class="recommendation-title">{rec['title']}</div>
                    <p>{rec['content']}</p>
                </div>
            """
        
        # Final verdict
        report += f"""
                <div class="insight-box">
                    <h4>Final Verdict</h4>
                    <p>Based on comprehensive analysis of {len(comparison_df)} metrics across {len(categories)} categories, 
                    <span class="{'player1-color' if overall_diff > 0 else 'player2-color'}">{better_player}</span> 
                    provides more overall value for most teams.</p>
                    
                    <p>The similarity score of {similarity_score:.2f} suggests these players are 
                    {similarity_text.lower()}.</p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #888; font-size: 0.9em;">
                <p>NBA Comparison Tool 2025 • Advanced Analytics Platform • Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
            </div>
        </div>
        """
        
        return report
    except Exception as e:
        st.error(f"Report generation error: {str(e)}")
        return "Could not generate full report."

def display_player_header(player, df):
    """Display player header with image, stats, and team logo"""
    try:
        player_data = df[df['player'] == player].iloc[0]
        player_id = player_data.get('player_id', '')
        team_abbrev = player_data.get('team', '')
        
        # Get team colors for styling
        team_colors = get_team_colors(team_abbrev)
        primary_color = team_colors[0] if team_colors else '#4caf50'
        secondary_color = team_colors[1] if team_colors and len(team_colors) > 1 else '#ffffff'
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Display player headshot with error handling
            headshot = get_player_headshot(player_id)
            if headshot:
                img_str = image_to_base64(headshot)
                st.markdown(
                    f'<div style="display: flex; justify-content: center;">'
                    f'<img src="data:image/png;base64,{img_str}" style="border-radius:10px;width:150px;border:2px solid {primary_color};">'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="width:150px;height:150px;background-color:#1e2130;border-radius:10px;'
                    f'display:flex;align-items:center;justify-content:center;margin:0 auto;border:2px solid {primary_color};">'
                    f'<span style="color:#666;">No Image</span></div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown(f"""
                <div style="padding: 10px;">
                    <h2 style='margin-bottom: 5px;'>{player}</h2>
                    <p style='margin-top: 0; color: #aaa;'>
                        {player_data.get('position_detail', '')} | {team_abbrev} | Age: {player_data.get('age', 'N/A')}
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px;">
                        <div>
                            <strong>PPG:</strong> {player_data.get('points_per_game', 0):.1f}<br>
                            <strong>RPG:</strong> {player_data.get('rebounds_per_game', 0):.1f}
                        </div>
                        <div>
                            <strong>APG:</strong> {player_data.get('assists_per_game', 0):.1f}<br>
                            <strong>MPG:</strong> {player_data.get('minutes_per_game', 0):.1f}
                        </div>
                    </div>
                    <div style="margin-top: 10px;">
                        <small>
                            <strong>PER:</strong> {player_data.get('per', 0):.1f} | 
                            <strong>TS%:</strong> {player_data.get('ts_pct', 0):.3f} | 
                            <strong>USG%:</strong> {player_data.get('usg_pct', 0):.1f}
                        </small>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Display team logo with centered alignment
            logo = get_team_logo(team_abbrev)
            if logo:
                img_str = image_to_base64(logo)
                st.markdown(
                    f'<div style="display: flex; justify-content: center; align-items: center; height: 100%;">'
                    f'<img src="data:image/png;base64,{img_str}" style="max-width:100px;max-height:100px;">'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.write("")
    
    except Exception as e:
        st.error(f"Error displaying header for {player}: {str(e)}")

def display_player_stats(player, df):
    """Display detailed stats for a player"""
    try:
        player_data = df[df['player'] == player].iloc[0]
        team_abbrev = player_data.get('team', '')
        team_colors = get_team_colors(team_abbrev)
        primary_color = team_colors[0] if team_colors else '#4caf50'
        
        # Basic info
        st.markdown(f"""
            <div style="background-color: #1e2130; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid {primary_color};">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                    <div>
                        <strong>Team:</strong> {team_abbrev}<br>
                        <strong>Position:</strong> {player_data.get('position_detail', 'N/A')}
                    </div>
                    <div>
                        <strong>Age:</strong> {player_data.get('age', 'N/A')}<br>
                        <strong>Height:</strong> {player_data.get('height', 'N/A')}
                    </div>
                    <div>
                        <strong>Weight:</strong> {player_data.get('weight', 'N/A')}<br>
                        <strong>Experience:</strong> {player_data.get('experience', 'N/A')}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Stats in tabs
        tab1, tab2, tab3 = st.tabs(["Per Game", "Advanced", "Shooting"])
        
        with tab1:
            st.markdown("### Per Game Stats")
            per_game_stats = [
                ('Points', 'points_per_game'),
                ('Rebounds', 'rebounds_per_game'),
                ('Assists', 'assists_per_game'),
                ('Steals', 'steals_per_game'),
                ('Blocks', 'blocks_per_game'),
                ('Minutes', 'minutes_per_game'),
                ('FGA', 'fga_per_game'),
                ('3PA', '3pa_per_game'),
                ('FTA', 'fta_per_game')
            ]
            
            cols = st.columns(3)
            for i, (label, stat) in enumerate(per_game_stats):
                with cols[i % 3]:
                    st.metric(label, f"{player_data.get(stat, 0):.1f}")
        
        with tab2:
            st.markdown("### Advanced Stats")
            advanced_stats = [
                ('PER', 'per'),
                ('TS%', 'ts_pct'),
                ('eFG%', 'efg_pct'),
                ('USG%', 'usg_pct'),
                ('AST/TO', 'ast_to'),
                ('STL+BLK', 'stl_blk'),
                ('Value', 'player_value')
            ]
            
            cols = st.columns(3)
            for i, (label, stat) in enumerate(advanced_stats):
                with cols[i % 3]:
                    if '%' in label:
                        st.metric(label, f"{player_data.get(stat, 0):.1%}")
                    else:
                        st.metric(label, f"{player_data.get(stat, 0):.1f}")
        
        with tab3:
            st.markdown("### Shooting Percentages")
            shooting_stats = [
                ('FG%', 'fg_pct'),
                ('3P%', '3p_pct'),
                ('FT%', 'ft_pct'),
                ('2P%', '2p_pct')
            ]
            
            cols = st.columns(4)
            for i, (label, stat) in enumerate(shooting_stats):
                with cols[i]:
                    st.metric(label, f"{player_data.get(stat, 0):.1%}")
    
    except Exception as e:
        st.error(f"Error displaying stats: {str(e)}")

def main():
    """Main application with enhanced UI and performance optimizations"""
    # Page configuration with improved metadata
    st.set_page_config(
        page_title="NBA Comparison Tool 2025",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': "https://github.com/your-repo/issues",
            'About': "# NBA Player Comparison Tool\nAdvanced analytics for player evaluation"
        }
    )
    
    # Custom CSS with improved styling
    st.markdown("""
        <style>
            /* Main page styling */
            .main {
                background-color: #0e1117;
                color: #ffffff;
            }
            
            /* Improved metric cards */
            .stMetric {
                background-color: #1e2130;
                border-radius: 10px;
                padding: 15px;
                margin: 5px 0;
                border-left: 4px solid #1e88e5;
            }
            
            /* Better progress bars */
            .stProgress > div > div > div > div {
                background-color: #1e88e5;
            }
            
            /* Enhanced tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                padding: 0 20px;
                background-color: #1e2130;
                border-radius: 10px 10px 0 0;
                border: none;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #1e88e5;
                color: white;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #2d303e;
            }
            
            /* Improved dataframes */
            .stDataFrame {
                border-radius: 10px;
                border: 1px solid #2d303e;
            }
            
            /* Better select boxes */
            .stSelectbox > div > div {
                background-color: #1e2130;
                border-radius: 6px;
                border: 1px solid #2d303e;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #1e2130;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #1e88e5;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #1565c0;
            }
            
            /* Loading spinner color */
            .stSpinner > div > div {
                border-top-color: #1e88e5;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # App header with improved layout
    st.title("🏀 NBA Comparison Tool 2025")
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    st.markdown(f"""
        <div style="color: #bbbbbb; margin-bottom: 30px;">
            Advanced player analytics and comparison tool for the 2024-25 NBA season.<br>
            <small>Data source: {DATA_FILE} | Last updated: {current_date}</small>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data with progress indicator and error handling
    with st.spinner("Loading NBA statistics..."):
        try:
            df = load_data()
            if df.empty:
                st.error("Failed to load data. Please check the data file and try again.")
                st.stop()
        except Exception as e:
            st.error(f"Critical error loading data: {str(e)}")
            st.stop()
    
    # Sidebar configuration with improved filters
    st.sidebar.header("🔍 Player Selection")
    
    # Team filter with search
    teams = sorted(df['team'].unique()) if 'team' in df.columns else []
    selected_team = st.sidebar.selectbox(
        "Filter by Team", 
        ['All Teams'] + teams, 
        index=0,
        help="Filter players by team"
    )
    
    # Position filter with multi-select
    positions = sorted(df['position_detail'].unique()) if 'position_detail' in df.columns else []
    selected_position = st.sidebar.selectbox(
        "Filter by Position", 
        ['All Positions'] + positions, 
        index=0,
        help="Filter players by position"
    )
    
    # Apply filters with performance optimization
    filtered_df = df.copy()
    if selected_team != 'All Teams':
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
    if selected_position != 'All Positions':
        filtered_df = filtered_df[filtered_df['position_detail'] == selected_position]
    
    players = sorted(filtered_df['player'].unique())
    
    if not players:
        st.sidebar.warning("No players match the selected filters")
        st.stop()
    
    # Player selection with improved layout
    st.sidebar.markdown("### Select Players to Compare")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        player1 = st.selectbox(
            "Player 1", 
            players, 
            index=0,
            key="player1_select",
            help="Select first player for comparison"
        )
    with col2:
        remaining_players = [p for p in players if p != player1]
        default_index = min(1, len(remaining_players)-1) if len(remaining_players) > 1 else 0
        player2 = st.selectbox(
            "Player 2", 
            remaining_players, 
            index=default_index,
            key="player2_select",
            help="Select second player for comparison"
        )
    
    # Add compare button with loading state
    if st.sidebar.button("🚀 Compare Players", use_container_width=True, type="primary"):
        with st.spinner("Generating comparison..."):
            time.sleep(0.5)  # Simulate processing for better UX
            st.rerun()
    
    # Add quick player search in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔎 Quick Player Search")
    search_term = st.sidebar.text_input("Search by name", "")
    if search_term:
        search_results = [p for p in players if search_term.lower() in p.lower()]
        if search_results:
            st.sidebar.markdown("**Search Results**")
            for player in search_results[:5]:  # Limit to top 5 results
                if st.sidebar.button(player, use_container_width=True):
                    # Update the player selection
                    if player != player1 and player != player2:
                        if len(remaining_players) > 1:
                            player2 = player
                        else:
                            player1 = player
                    st.rerun()
        else:
            st.sidebar.warning("No players found matching your search")
    
    # Main content with player comparison
    if player1 and player2:
        # Generate comparison data with error handling
        try:
            with st.spinner("Calculating player comparison..."):
                comparison_df = create_comparison_df(player1, player2, df)
                similarity_scores = calculate_similarity(player1, player2, df)
                
                if comparison_df.empty:
                    st.error("Could not generate comparison data")
                    st.stop()
                
                # Get team colors for styling
                p1_team = df[df['player'] == player1]['team'].iloc[0] if 'team' in df.columns else ''
                p2_team = df[df['player'] == player2]['team'].iloc[0] if 'team' in df.columns else ''
                
                team1_colors = get_team_colors(p1_team)
                team2_colors = get_team_colors(p2_team)
                
                player1_color = team1_colors[0] if team1_colors else '#4caf50'
                player2_color = team2_colors[0] if team2_colors else '#f44336'
                
                # Player headers with improved spacing
                st.markdown("---")
                display_player_header(player1, df)
                st.markdown("---")
                display_player_header(player2, df)
                st.markdown("---")
                
                # Main comparison section with tabs
                st.markdown("## 📊 Statistical Comparison")
                
                # Use tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Key Metrics", "📊 Visualization", "📉 Detailed Breakdown", "🌐 Similarity Map"])
                
                with tab1:
                    # Create a styled DataFrame with sorting options
                    st.markdown("### 🏆 Performance Metrics Comparison")
                    
                    # Add sorting options
                    col1, col2 = st.columns(2)
                    with col1:
                        sort_by = st.selectbox(
                            "Sort by", 
                            ['Category', 'Difference', 'Metric'], 
                            index=0,
                            key="sort_select"
                        )
                    with col2:
                        sort_order = st.selectbox(
                            "Order", 
                            ['Descending', 'Ascending'], 
                            index=0,
                            key="order_select"
                        )
                    
                    # Sort the DataFrame
                    if sort_by == 'Category':
                        sorted_df = comparison_df.sort_values(
                            'category', 
                            ascending=(sort_order == 'Ascending')
                        )
                    elif sort_by == 'Difference':
                        sorted_df = comparison_df.sort_values(
                            'difference', 
                            ascending=(sort_order == 'Ascending')
                        )
                    else:
                        sorted_df = comparison_df.sort_values(
                            'metric', 
                            ascending=(sort_order == 'Ascending')
                        )
                    
                    # Display styled DataFrame
                    try:
                        # Create pivot table for better display
                        pivot_df = sorted_df.pivot(
                            index='metric', 
                            columns='category', 
                            values=['player1', 'player2', 'difference']
                        )
                        
                        # Format the styled DataFrame
                        def color_diff(val):
                            if isinstance(val, (int, float)):
                                if val > 0:
                                    return f"color: {player1_color}; font-weight: bold"  # Green for positive
                                elif val < 0:
                                    return f"color: {player2_color}; font-weight: bold"  # Red for negative
                            return ""
                        
                        st.dataframe(
                            pivot_df.style
                            .format("{:.2f}")
                            .map(color_diff, subset=pd.IndexSlice[:, 'difference'])
                            .set_properties(**{'background-color': '#1e2130', 'color': 'white'})
                            .set_table_styles([{
                                'selector': 'thead th',
                                'props': [('background-color', '#1e88e5'), ('color', 'white')]
                            }]),
                            height=600,
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Could not format comparison table: {str(e)}")
                        st.dataframe(sorted_df, height=600, use_container_width=True)
                
                with tab2:
                    # Visualization tab with multiple charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar chart
                        st.markdown("### 📊 Skills Radar Chart")
                        radar_fig = create_radar_chart(player1, player2, comparison_df, player1_color, player2_color)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with col2:
                        # Similarity gauge
                        st.markdown("### 🔄 Player Similarity")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=similarity_scores['combined'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Player Similarity Score", 'font': {'size': 20}},
                            gauge={
                                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': player1_color},
                                'bgcolor': "black",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 0.4], 'color': "red"},
                                    {'range': [0.4, 0.7], 'color': "yellow"},
                                    {'range': [0.7, 1], 'color': "green"}],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': similarity_scores['combined']}
                            },
                            delta={'reference': 0.5, 'increasing': {'color': "green"}},
                            number={'font': {'size': 40}}
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Difference bar chart
                        st.markdown("### 📉 Biggest Differences")
                        diff_df = comparison_df[['metric', 'difference']].set_index('metric')
                        diff_df = diff_df.sort_values('difference', ascending=False)
                        fig = px.bar(
                            diff_df.head(10),
                            orientation='h',
                            title=f"Top 10 Metrics with Biggest Differences",
                            labels={'value': 'Difference', 'index': 'Metric'},
                            color='difference',
                            color_continuous_scale=[player2_color, '#888888', player1_color],
                            range_color=[diff_df['difference'].min(), diff_df['difference'].max()]
                        )
                        fig.update_layout(
                            template='plotly_dark',
                            yaxis={'categoryorder': 'total ascending'},
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Detailed breakdown by category
                    st.markdown("### 🔍 Metric-by-Metric Comparison")
                    
                    # Group by category
                    categories = comparison_df['category'].unique()
                    for category in categories:
                        cat_df = comparison_df[comparison_df['category'] == category]
                        
                        with st.expander(f"📌 {category} ({len(cat_df)} metrics)"):
                            cols = st.columns(3)
                            
                            for idx, row in cat_df.iterrows():
                                with cols[idx % 3]:
                                    metric = row['metric']
                                    p1_val = row['player1']
                                    p2_val = row['player2']
                                    diff = row['difference']
                                    percent_diff = row['percent_difference']
                                    
                                    st.metric(
                                        label=metric.replace('_', ' ').title(),
                                        value=f"{p1_val:.2f}",
                                        delta=f"{p2_val:.2f} ({diff:+.2f}, {percent_diff:+.1f}%)",
                                        delta_color="normal",
                                        help=f"{player1}: {p1_val:.2f} | {player2}: {p2_val:.2f}"
                                    )
                
                with tab4:
                    # Player similarity landscape
                    st.markdown("### 🌐 Player Similarity Landscape")
                    st.markdown("This visualization shows how players cluster based on their statistical profiles. Closer players are more similar.")
                    
                    similarity_fig = create_player_similarity_plot(player1, player2, df)
                    if similarity_fig:
                        st.plotly_chart(similarity_fig, use_container_width=True)
                    else:
                        st.warning("Could not generate similarity plot")
                
                # Generate and display comprehensive report
                st.markdown("---")
                st.markdown("## 📝 Advanced Analytical Report")
                with st.expander("View Full Analysis Report", expanded=True):
                    report = generate_comprehensive_report(player1, player2, comparison_df, similarity_scores, df)
                    st.markdown(report, unsafe_allow_html=True)
                
                # Strengths and weaknesses analysis
                st.markdown("---")
                st.markdown("## ✅ Strengths and Weaknesses Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### 🟢 {player1}'s Strengths")
                    strengths = comparison_df[comparison_df['difference'] > 0].sort_values('difference', ascending=False)
                    
                    # Group by category for better organization
                    for category in strengths['category'].unique():
                        cat_strengths = strengths[strengths['category'] == category]
                        if not cat_strengths.empty:
                            with st.expander(f"{category} ({len(cat_strengths)} metrics)"):
                                for _, row in cat_strengths.iterrows():
                                    st.success(
                                        f"**{row['metric'].replace('_', ' ').title()}**: "
                                        f"+{row['difference']:.2f} "
                                        f"({row['player1']:.2f} vs {row['player2']:.2f})"
                                    )
                
                with col2:
                    st.markdown(f"### 🔴 {player2}'s Strengths")
                    weaknesses = comparison_df[comparison_df['difference'] < 0].sort_values('difference')
                    
                    # Group by category for better organization
                    for category in weaknesses['category'].unique():
                        cat_weaknesses = weaknesses[weaknesses['category'] == category]
                        if not cat_weaknesses.empty:
                            with st.expander(f"{category} ({len(cat_weaknesses)} metrics)"):
                                for _, row in cat_weaknesses.iterrows():
                                    st.error(
                                        f"**{row['metric'].replace('_', ' ').title()}**: "
                                        f"+{abs(row['difference']):.2f} "
                                        f"({row['player2']:.2f} vs {row['player1']:.2f})"
                                    )
        
        except Exception as e:
            st.error(f"Error generating comparison: {str(e)}")
            st.stop()
    
    # Footer with improved styling
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align:center;color:#888;font-size:0.9em;padding:20px;">
            <p>NBA Comparison Tool 2025 • Advanced Analytics Platform</p>
            <p><small>Data updated: {current_date} | Version: 2.1.0</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()