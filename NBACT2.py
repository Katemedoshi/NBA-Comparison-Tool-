import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
import warnings
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NBA Player Comparison Tool",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
HEADSHOT_BASE_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/"
TEAM_COLORS = {
    'ATL': '#E03A3E', 'BOS': '#007A33', 'BRK': '#000000', 'CHA': '#1D1160',
    'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240',
    'DET': '#C8102E', 'GSW': '#1D428A', 'HOU': '#CE1141', 'IND': '#002D62',
    'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E',
    'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160',
    'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141',
    'UTA': '#002B5C', 'WAS': '#002B5C'
}

# Mock player data for demonstration
MOCK_PLAYERS = {
    'LeBron James': {
        'id': 2544, 'name': 'LeBron James', 'position': 'F', 'team': 'LAL',
        'height': '6-9', 'weight': '250', 'age': 39,
        'stats': {'ppg': 25.7, 'rpg': 7.3, 'apg': 8.3, 'spg': 1.3, 'bpg': 0.6,
                 'fg_pct': 0.540, '3p_pct': 0.410, 'ft_pct': 0.740, 'mpg': 35.5}
    },
    'Stephen Curry': {
        'id': 201939, 'name': 'Stephen Curry', 'position': 'G', 'team': 'GSW',
        'height': '6-2', 'weight': '185', 'age': 36,
        'stats': {'ppg': 26.4, 'rpg': 4.5, 'apg': 5.1, 'spg': 0.7, 'bpg': 0.3,
                 'fg_pct': 0.450, '3p_pct': 0.407, 'ft_pct': 0.923, 'mpg': 32.7}
    },
    'Giannis Antetokounmpo': {
        'id': 203507, 'name': 'Giannis Antetokounmpo', 'position': 'F', 'team': 'MIL',
        'height': '6-11', 'weight': '243', 'age': 29,
        'stats': {'ppg': 30.4, 'rpg': 11.5, 'apg': 6.5, 'spg': 1.2, 'bpg': 1.1,
                 'fg_pct': 0.611, '3p_pct': 0.275, 'ft_pct': 0.656, 'mpg': 35.0}
    },
    'Nikola Jokic': {
        'id': 203999, 'name': 'Nikola Jokic', 'position': 'C', 'team': 'DEN',
        'height': '6-11', 'weight': '284', 'age': 29,
        'stats': {'ppg': 26.4, 'rpg': 12.4, 'apg': 9.0, 'spg': 1.4, 'bpg': 0.9,
                 'fg_pct': 0.583, '3p_pct': 0.357, 'ft_pct': 0.821, 'mpg': 34.6}
    },
    'Luka Doncic': {
        'id': 1629029, 'name': 'Luka Doncic', 'position': 'G', 'team': 'DAL',
        'height': '6-7', 'weight': '230', 'age': 25,
        'stats': {'ppg': 33.9, 'rpg': 9.2, 'apg': 9.8, 'spg': 1.4, 'bpg': 0.5,
                 'fg_pct': 0.487, '3p_pct': 0.382, 'ft_pct': 0.786, 'mpg': 37.5}
    },
    'Jayson Tatum': {
        'id': 1628369, 'name': 'Jayson Tatum', 'position': 'F', 'team': 'BOS',
        'height': '6-8', 'weight': '210', 'age': 26,
        'stats': {'ppg': 26.9, 'rpg': 8.1, 'apg': 4.9, 'spg': 1.0, 'bpg': 0.6,
                 'fg_pct': 0.471, '3p_pct': 0.375, 'ft_pct': 0.834, 'mpg': 35.7}
    },
    'Kevin Durant': {
        'id': 201142, 'name': 'Kevin Durant', 'position': 'F', 'team': 'PHX',
        'height': '6-11', 'weight': '240', 'age': 35,
        'stats': {'ppg': 27.1, 'rpg': 6.6, 'apg': 5.0, 'spg': 0.9, 'bpg': 1.2,
                 'fg_pct': 0.525, '3p_pct': 0.416, 'ft_pct': 0.885, 'mpg': 37.2}
    },
    'Kawhi Leonard': {
        'id': 202695, 'name': 'Kawhi Leonard', 'position': 'F', 'team': 'LAC',
        'height': '6-7', 'weight': '225', 'age': 32,
        'stats': {'ppg': 23.8, 'rpg': 6.1, 'apg': 3.4, 'spg': 1.7, 'bpg': 0.5,
                 'fg_pct': 0.525, '3p_pct': 0.417, 'ft_pct': 0.884, 'mpg': 34.3}
    }
}

@st.cache_data(ttl=3600)
def search_players(query):
    """Search for players based on query"""
    if not query:
        return []
    
    query_lower = query.lower()
    matches = []
    
    for name, data in MOCK_PLAYERS.items():
        if query_lower in name.lower():
            matches.append(data)
    
    return matches[:10]  # Return top 10 matches

def get_player_image(player_id, player_name):
    """Get player headshot image"""
    try:
        # Try to construct URL from player name
        formatted_name = player_name.lower().replace(' ', '-')
        url = f"{HEADSHOT_BASE_URL}{formatted_name}.png"
        
        # For mock data, return placeholder with team color
        team = MOCK_PLAYERS.get(player_name, {}).get('team', 'NBA')
        color = TEAM_COLORS.get(team, '#4CAF50')
        
        # Create a placeholder image with initials
        initials = ''.join([word[0] for word in player_name.split()[:2]]).upper()
        
        # Return HTML for placeholder
        return f"""
        <div style="width: 120px; height: 120px; background-color: {color}; 
                    border-radius: 50%; display: flex; align-items: center; 
                    justify-content: center; color: white; font-size: 36px;
                    font-weight: bold; margin: 0 auto;">
            {initials}
        </div>
        """
    except:
        return ""

def calculate_advanced_stats(player_stats):
    """Calculate advanced statistics"""
    stats = player_stats.copy()
    
    # Calculate PER (simplified version)
    stats['per'] = (
        stats['ppg'] * 0.25 +
        stats['rpg'] * 0.15 +
        stats['apg'] * 0.15 +
        stats['spg'] * 0.1 +
        stats['bpg'] * 0.1 +
        stats['fg_pct'] * 100 * 0.15 +
        (1 - (stats['fg_pct'] + stats['ft_pct']) / 2) * 10  # Adjust for inefficiency
    )
    
    # Calculate TS%
    stats['ts_pct'] = stats['ppg'] / (2 * (stats['ppg'] / stats['fg_pct'] if stats['fg_pct'] > 0 else 1))
    
    # Calculate Value Score
    stats['value'] = (
        stats['ppg'] * 0.3 +
        stats['rpg'] * 0.2 +
        stats['apg'] * 0.2 +
        stats['spg'] * 0.1 +
        stats['bpg'] * 0.1 +
        stats['fg_pct'] * 100 * 0.1
    )
    
    return stats

def create_comparison_data(player1_data, player2_data):
    """Create comparison data structure"""
    p1_stats = player1_data['stats']
    p2_stats = player2_data['stats']
    
    # Basic stats comparison
    basic_stats = {
        'Points': {'player1': p1_stats['ppg'], 'player2': p2_stats['ppg'], 'unit': 'PPG'},
        'Rebounds': {'player1': p1_stats['rpg'], 'player2': p2_stats['rpg'], 'unit': 'RPG'},
        'Assists': {'player1': p1_stats['apg'], 'player2': p2_stats['apg'], 'unit': 'APG'},
        'Steals': {'player1': p1_stats['spg'], 'player2': p2_stats['spg'], 'unit': 'SPG'},
        'Blocks': {'player1': p1_stats['bpg'], 'player2': p2_stats['bpg'], 'unit': 'BPG'},
        'Minutes': {'player1': p1_stats['mpg'], 'player2': p2_stats['mpg'], 'unit': 'MPG'}
    }
    
    # Shooting stats
    shooting_stats = {
        'FG%': {'player1': p1_stats['fg_pct'] * 100, 'player2': p2_stats['fg_pct'] * 100, 'unit': '%'},
        '3P%': {'player1': p1_stats['3p_pct'] * 100, 'player2': p2_stats['3p_pct'] * 100, 'unit': '%'},
        'FT%': {'player1': p1_stats['ft_pct'] * 100, 'player2': p2_stats['ft_pct'] * 100, 'unit': '%'}
    }
    
    # Advanced stats
    p1_advanced = calculate_advanced_stats(p1_stats)
    p2_advanced = calculate_advanced_stats(p2_stats)
    
    advanced_stats = {
        'PER': {'player1': p1_advanced['per'], 'player2': p2_advanced['per'], 'unit': ''},
        'TS%': {'player1': p1_advanced['ts_pct'] * 100, 'player2': p2_advanced['ts_pct'] * 100, 'unit': '%'},
        'Value': {'player1': p1_advanced['value'], 'player2': p2_advanced['value'], 'unit': ''}
    }
    
    return {
        'basic': basic_stats,
        'shooting': shooting_stats,
        'advanced': advanced_stats,
        'player1_advanced': p1_advanced,
        'player2_advanced': p2_advanced
    }

def generate_insights(comparison_data, player1_data, player2_data):
    """Generate intelligent insights"""
    insights = []
    
    p1_name = player1_data['name']
    p2_name = player2_data['name']
    p1_stats = player1_data['stats']
    p2_stats = player2_data['stats']
    
    # Points insight
    ppg_diff = p1_stats['ppg'] - p2_stats['ppg']
    if abs(ppg_diff) >= 5:
        better_scorer = p1_name if ppg_diff > 0 else p2_name
        insights.append(f"**Scoring**: {better_scorer} averages **{abs(ppg_diff):.1f} more points per game**")
    elif abs(ppg_diff) > 0:
        better_scorer = p1_name if ppg_diff > 0 else p2_name
        insights.append(f"**Scoring**: {better_scorer} scores **{abs(ppg_diff):.1f} more PPG**")
    
    # Rebounds insight
    rpg_diff = p1_stats['rpg'] - p2_stats['rpg']
    if abs(rpg_diff) >= 3:
        better_rebounder = p1_name if rpg_diff > 0 else p2_name
        insights.append(f"**Rebounding**: {better_rebounder} grabs **{abs(rpg_diff):.1f} more rebounds per game**")
    
    # Assists insight
    apg_diff = p1_stats['apg'] - p2_stats['apg']
    if abs(apg_diff) >= 2:
        better_playmaker = p1_name if apg_diff > 0 else p2_name
        insights.append(f"**Playmaking**: {better_playmaker} creates **{abs(apg_diff):.1f} more assists per game**")
    
    # Efficiency insight
    fg_diff = (p1_stats['fg_pct'] - p2_stats['fg_pct']) * 100
    if abs(fg_diff) >= 3:
        more_efficient = p1_name if fg_diff > 0 else p2_name
        insights.append(f"**Efficiency**: {more_efficient} shoots **{abs(fg_diff):.1f}% better from the field**")
    
    # 3-point insight
    three_pt_diff = (p1_stats['3p_pct'] - p2_stats['3p_pct']) * 100
    if abs(three_pt_diff) >= 5:
        better_shooter = p1_name if three_pt_diff > 0 else p2_name
        insights.append(f"**3-Point**: {better_shooter} is **{abs(three_pt_diff):.1f}% better from three**")
    
    # Defensive insight
    stl_blk_diff = (p1_stats['spg'] + p1_stats['bpg']) - (p2_stats['spg'] + p2_stats['bpg'])
    if abs(stl_blk_diff) >= 0.8:
        better_defender = p1_name if stl_blk_diff > 0 else p2_name
        insights.append(f"**Defense**: {better_defender} makes **{abs(stl_blk_diff):.1f} more defensive plays per game**")
    
    # Minutes insight
    mpg_diff = p1_stats['mpg'] - p2_stats['mpg']
    if abs(mpg_diff) >= 5:
        more_playing = p1_name if mpg_diff > 0 else p2_name
        insights.append(f"**Playing Time**: {more_playing} plays **{abs(mpg_diff):.1f} more minutes per game**")
    
    # Value insight
    p1_value = comparison_data['player1_advanced']['value']
    p2_value = comparison_data['player2_advanced']['value']
    value_diff = p1_value - p2_value
    if abs(value_diff) >= 2:
        more_valuable = p1_name if value_diff > 0 else p2_name
        insights.append(f"**Overall Value**: {more_valuable} provides **{abs(value_diff):.1f} more value** based on comprehensive metrics")
    
    if not insights:
        insights.append("These players have remarkably similar statistical profiles across all major categories")
    
    return insights

def create_radar_chart(player1_data, player2_data):
    """Create radar chart for skill comparison"""
    categories = ['Scoring', 'Rebounding', 'Playmaking', 'Defense', 'Efficiency']
    
    # Normalize values for radar chart (0-100 scale)
    p1_stats = player1_data['stats']
    p2_stats = player2_data['stats']
    
    # Scoring (based on PPG and efficiency)
    p1_scoring = (p1_stats['ppg'] / 40 * 100) * 0.7 + (p1_stats['fg_pct'] * 100) * 0.3
    p2_scoring = (p2_stats['ppg'] / 40 * 100) * 0.7 + (p2_stats['fg_pct'] * 100) * 0.3
    
    # Rebounding
    p1_rebounding = (p1_stats['rpg'] / 15 * 100)
    p2_rebounding = (p2_stats['rpg'] / 15 * 100)
    
    # Playmaking
    p1_playmaking = (p1_stats['apg'] / 12 * 100)
    p2_playmaking = (p2_stats['apg'] / 12 * 100)
    
    # Defense
    p1_defense = ((p1_stats['spg'] + p1_stats['bpg']) / 4 * 100)
    p2_defense = ((p2_stats['spg'] + p2_stats['bpg']) / 4 * 100)
    
    # Efficiency (based on shooting percentages)
    p1_efficiency = (p1_stats['fg_pct'] + p1_stats['3p_pct'] + p1_stats['ft_pct']) / 3 * 100
    p2_efficiency = (p2_stats['fg_pct'] + p2_stats['3p_pct'] + p2_stats['ft_pct']) / 3 * 100
    
    p1_values = [p1_scoring, p1_rebounding, p1_playmaking, p1_defense, p1_efficiency]
    p2_values = [p2_scoring, p2_rebounding, p2_playmaking, p2_defense, p2_efficiency]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=p1_values + [p1_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=player1_data['name'],
        line=dict(color='#4CAF50', width=2),
        fillcolor='rgba(76, 175, 80, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=p2_values + [p2_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=player2_data['name'],
        line=dict(color='#F44336', width=2),
        fillcolor='rgba(244, 67, 54, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def display_player_header(player_data, side="left"):
    """Display player header with stats"""
    team_color = TEAM_COLORS.get(player_data['team'], '#4CAF50')
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {team_color}20 0%, {team_color}10 100%);
                border-left: 5px solid {team_color};
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;">
        <div style="display: flex; align-items: center;">
            <div style="flex: 1;">
                <h2 style="margin: 0; color: #333;">{player_data['name']}</h2>
                <p style="color: #666; margin: 5px 0;">
                    {player_data['position']} • {player_data['team']} • Age: {player_data['age']}<br>
                    {player_data['height']} • {player_data['weight']} lbs
                </p>
            </div>
            <div style="width: 80px; height: 80px; background-color: {team_color}; 
                        border-radius: 50%; display: flex; align-items: center; 
                        justify-content: center; color: white; font-size: 24px;
                        font-weight: bold; margin-left: 20px;">
                {player_data['name'].split()[0][0]}{player_data['name'].split()[-1][0]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_comparison_table(comparison_data, player1_name, player2_name):
    """Display comparison table"""
    st.markdown("### 📊 Statistical Comparison")
    
    # Create DataFrame for basic stats
    basic_df = pd.DataFrame([
        {
            'Stat': stat,
            player1_name: f"{data['player1']:.1f}{data['unit']}",
            player2_name: f"{data['player2']:.1f}{data['unit']}",
            'Difference': f"{data['player1'] - data['player2']:+.1f}{data['unit']}"
        }
        for stat, data in comparison_data['basic'].items()
    ])
    
    # Create DataFrame for shooting stats
    shooting_df = pd.DataFrame([
        {
            'Stat': stat,
            player1_name: f"{data['player1']:.1f}{data['unit']}",
            player2_name: f"{data['player2']:.1f}{data['unit']}",
            'Difference': f"{data['player1'] - data['player2']:+.1f}{data['unit']}"
        }
        for stat, data in comparison_data['shooting'].items()
    ])
    
    # Create DataFrame for advanced stats
    advanced_df = pd.DataFrame([
        {
            'Stat': stat,
            player1_name: f"{data['player1']:.1f}{data['unit']}",
            player2_name: f"{data['player2']:.1f}{data['unit']}",
            'Difference': f"{data['player1'] - data['player2']:+.1f}{data['unit']}"
        }
        for stat, data in comparison_data['advanced'].items()
    ])
    
    # Display tables in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Basic Stats", "🎯 Shooting", "⚡ Advanced"])
    
    with tab1:
        st.dataframe(basic_df.set_index('Stat'), use_container_width=True)
    
    with tab2:
        st.dataframe(shooting_df.set_index('Stat'), use_container_width=True)
    
    with tab3:
        st.dataframe(advanced_df.set_index('Stat'), use_container_width=True)

def main():
    """Main application"""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #4CAF50;
    }
    .stat-card2 {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #F44336;
    }
    .insight-box {
        background: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .vs-container {
        text-align: center;
        padding: 40px 0;
    }
    .vs-text {
        font-size: 48px;
        font-weight: bold;
        color: #666;
        background: white;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4090 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏀 NBA Player Comparison Tool 2024</h1>
        <h3>Compare NBA Players with Advanced Analytics & Insights</h3>
        <p style="opacity: 0.9;">Real-time stats • Smart analysis • Visual comparisons</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔍 Player Selection")
    
    # Get all available players
    available_players = list(MOCK_PLAYERS.keys())
    
    # Player 1 selection
    player1_name = st.sidebar.selectbox(
        "Select Player 1:",
        options=available_players,
        index=0,
        key="player1_select"
    )
    
    # Player 2 selection (exclude player 1)
    player2_options = [p for p in available_players if p != player1_name]
    player2_name = st.sidebar.selectbox(
        "Select Player 2:",
        options=player2_options,
        index=min(1, len(player2_options)-1),
        key="player2_select"
    )
    
    # Season selector
    season = st.sidebar.selectbox(
        "Season:",
        ["2023-24", "2022-23", "2021-22"],
        index=0
    )
    
    # Comparison button
    compare_clicked = st.sidebar.button("🚀 Compare Players", type="primary", use_container_width=True)
    
    # Get player data
    player1_data = MOCK_PLAYERS.get(player1_name)
    player2_data = MOCK_PLAYERS.get(player2_name)
    
    if compare_clicked and player1_data and player2_data:
        # Generate comparison
        with st.spinner("Generating comparison..."):
            # Create comparison data
            comparison_data = create_comparison_data(player1_data, player2_data)
            
            # Generate insights
            insights = generate_insights(comparison_data, player1_data, player2_data)
            
            # Display player headers
            col1, col2, col3 = st.columns([1, 0.2, 1])
            
            with col1:
                display_player_header(player1_data)
            
            with col2:
                st.markdown("""
                <div class="vs-container">
                    <div class="vs-text">VS</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                display_player_header(player2_data)
            
            # Stats comparison
            st.markdown("---")
            display_comparison_table(comparison_data, player1_name, player2_name)
            
            # Key stats cards
            st.markdown("---")
            st.markdown("### 🏆 Key Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ppg_diff = player1_data['stats']['ppg'] - player2_data['stats']['ppg']
                color = "green" if ppg_diff > 0 else "red" if ppg_diff < 0 else "gray"
                st.metric(
                    label="Points Advantage",
                    value=f"{abs(ppg_diff):.1f} PPG",
                    delta=f"{player1_name if ppg_diff > 0 else player2_name} leads",
                    delta_color="normal" if color == "gray" else "off"
                )
            
            with col2:
                rpg_diff = player1_data['stats']['rpg'] - player2_data['stats']['rpg']
                color = "green" if rpg_diff > 0 else "red" if rpg_diff < 0 else "gray"
                st.metric(
                    label="Rebounds Advantage",
                    value=f"{abs(rpg_diff):.1f} RPG",
                    delta=f"{player1_name if rpg_diff > 0 else player2_name} leads",
                    delta_color="normal" if color == "gray" else "off"
                )
            
            with col3:
                apg_diff = player1_data['stats']['apg'] - player2_data['stats']['apg']
                color = "green" if apg_diff > 0 else "red" if apg_diff < 0 else "gray"
                st.metric(
                    label="Assists Advantage",
                    value=f"{abs(apg_diff):.1f} APG",
                    delta=f"{player1_name if apg_diff > 0 else player2_name} leads",
                    delta_color="normal" if color == "gray" else "off"
                )
            
            with col4:
                fg_diff = (player1_data['stats']['fg_pct'] - player2_data['stats']['fg_pct']) * 100
                color = "green" if fg_diff > 0 else "red" if fg_diff < 0 else "gray"
                st.metric(
                    label="FG% Advantage",
                    value=f"{abs(fg_diff):.1f}%",
                    delta=f"{player1_name if fg_diff > 0 else player2_name} leads",
                    delta_color="normal" if color == "gray" else "off"
                )
            
            # Insights section
            st.markdown("---")
            st.markdown("### 💡 Smart Insights")
            
            for insight in insights:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Skill Radar Chart")
                radar_fig = create_radar_chart(player1_data, player2_data)
                st.plotly_chart(radar_fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Shooting Comparison")
                
                shooting_data = pd.DataFrame({
                    'Category': ['Field Goal %', '3-Point %', 'Free Throw %'],
                    player1_name: [
                        player1_data['stats']['fg_pct'] * 100,
                        player1_data['stats']['3p_pct'] * 100,
                        player1_data['stats']['ft_pct'] * 100
                    ],
                    player2_name: [
                        player2_data['stats']['fg_pct'] * 100,
                        player2_data['stats']['3p_pct'] * 100,
                        player2_data['stats']['ft_pct'] * 100
                    ]
                })
                
                fig = px.bar(
                    shooting_data.melt(id_vars='Category'),
                    x='Category',
                    y='value',
                    color='variable',
                    barmode='group',
                    color_discrete_map={
                        player1_name: '#4CAF50',
                        player2_name: '#F44336'
                    }
                )
                fig.update_layout(
                    yaxis_title="Percentage (%)",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Player profiles
            st.markdown("---")
            st.markdown("### 👤 Player Profiles")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {player1_name}")
                profile_text = f"""
                **Position:** {player1_data['position']}
                
                **Team:** {player1_data['team']}
                
                **Age:** {player1_data['age']}
                
                **Size:** {player1_data['height']}, {player1_data['weight']} lbs
                
                **Playing Style:**
                - Primary scorer and playmaker
                - Elite athleticism and basketball IQ
                - Versatile defender
                - Clutch performer
                
                **Career Highlights:**
                - Multiple MVP awards
                - NBA Championships
                - All-Star selections
                - All-NBA Team honors
                """
                st.markdown(profile_text)
            
            with col2:
                st.markdown(f"#### {player2_name}")
                profile_text = f"""
                **Position:** {player2_data['position']}
                
                **Team:** {player2_data['team']}
                
                **Age:** {player2_data['age']}
                
                **Size:** {player2_data['height']}, {player2_data['weight']} lbs
                
                **Playing Style:**
                - Elite shooter and scorer
                - High basketball IQ
                - Offensive focal point
                - Team leader
                
                **Career Highlights:**
                - MVP awards
                - NBA Championships
                - All-Star selections
                - Scoring champion
                - Shooting records
                """
                st.markdown(profile_text)
            
            # Download report
            st.markdown("---")
            st.markdown("### 📄 Download Report")
            st.info("The comparison data can be downloaded as a CSV file.")
            
            # Create download DataFrame
            report_data = []
            for category, stats in comparison_data.items():
                if category not in ['player1_advanced', 'player2_advanced']:
                    for stat_name, stat_data in stats.items():
                        report_data.append({
                            'Category': category.capitalize(),
                            'Statistic': stat_name,
                            player1_name: stat_data['player1'],
                            player2_name: stat_data['player2'],
                            'Difference': stat_data['player1'] - stat_data['player2'],
                            'Unit': stat_data['unit']
                        })
            
            report_df = pd.DataFrame(report_data)
            
            # Convert to CSV
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Report",
                data=csv,
                file_name=f"nba_comparison_{player1_name.replace(' ', '_')}_vs_{player2_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 10px; margin-top: 20px;">
                <h2 style="color: #333;">Welcome to the NBA Comparison Tool! 🏀</h2>
                <p style="font-size: 16px; color: #666; line-height: 1.6;">
                    This tool allows you to compare NBA players with detailed statistics, 
                    advanced analytics, and intelligent insights. Select two players from 
                    the sidebar and click "Compare Players" to get started.
                </p>
                
                <h3 style="color: #444; margin-top: 30px;">✨ Features:</h3>
                <ul style="color: #666; font-size: 15px;">
                    <li><strong>Comprehensive Stats:</strong> Compare basic, shooting, and advanced metrics</li>
                    <li><strong>Smart Insights:</strong> Automated analysis of player strengths and weaknesses</li>
                    <li><strong>Visual Charts:</strong> Interactive radar charts and bar graphs</li>
                    <li><strong>Player Profiles:</strong> Detailed player information and career highlights</li>
                    <li><strong>Export Data:</strong> Download comparison reports as CSV</li>
                </ul>
                
                <h3 style="color: #444; margin-top: 30px;">🏆 Popular Comparisons:</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;">
                    <div style="background: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                        LeBron James vs Stephen Curry
                    </div>
                    <div style="background: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                        Giannis vs Jokic
                    </div>
                    <div style="background: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                        Luka Doncic vs Jayson Tatum
                    </div>
                    <div style="background: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                        Kevin Durant vs Kawhi Leonard
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 25px; border-radius: 10px; margin-top: 20px;">
                <h3 style="margin-top: 0;">📊 Quick Stats</h3>
                <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                    <div>
                        <h4 style="margin: 0; font-size: 24px;">8</h4>
                        <small>Players Available</small>
                    </div>
                    <div>
                        <h4 style="margin: 0; font-size: 24px;">28</h4>
                        <small>Stats Compared</small>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                    <div>
                        <h4 style="margin: 0; font-size: 24px;">5</h4>
                        <small>Visual Charts</small>
                    </div>
                    <div>
                        <h4 style="margin: 0; font-size: 24px;">10+</h4>
                        <small>Insights Generated</small>
                    </div>
                </div>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 10px; margin-top: 20px;
                        border-left: 4px solid #4CAF50;">
                <h4 style="color: #333; margin-top: 0;">⚡ How to Use:</h4>
                <ol style="color: #666; padding-left: 20px;">
                    <li>Select Player 1 from dropdown</li>
                    <li>Select Player 2 from dropdown</li>
                    <li>Choose season (optional)</li>
                    <li>Click "Compare Players"</li>
                    <li>Analyze results!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 14px; padding: 20px;">
            <p>NBA Player Comparison Tool • Data is for demonstration purposes • 
            Real NBA stats would be integrated in production version</p>
            <p>© 2024 NBA Comparison Tool • Made with Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
