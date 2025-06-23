# NBA-Comparison-Tool-

A comprehensive web application for comparing NBA players' statistics and performance metrics. This tool provides advanced analytics, visualizations, and insights to help basketball enthusiasts, analysts, and fantasy players make data-driven decisions.

# Features
Player Comparison: Compare any two NBA players across multiple statistical categories

Advanced Metrics: Includes PER, TS%, eFG%, and other advanced analytics

Interactive Visualizations: Radar charts, similarity scores, and difference plots

Comprehensive Reports: Detailed analysis with strengths/weaknesses breakdown

Team Branding: Player headshots and team logos with color theming

Similarity Analysis: Find players with similar statistical profiles

Responsive Design: Works on desktop and mobile devices

# Technologies Used
Python

Streamlit (Web Framework)

Pandas (Data Analysis)

Plotly (Interactive Visualizations)

Scikit-learn (Similarity Metrics)

# Data Requirements
The application requires an Excel file (NBA2025.xlsx) containing player statistics. The expected columns include:

Basic stats: points, rebounds, assists, etc.

Shooting percentages: FG%, 3P%, FT%

Advanced metrics: PER, TS%, eFG%

Player metadata: name, team, position, age

# Usage
Select two players to compare from the dropdown menus

Filter players by team or position if desired

Explore the different tabs:

Key Metrics: Statistical comparison table

Visualization: Radar chart and similarity gauge

Detailed Breakdown: Category-by-category analysis

Similarity Map: 2D visualization of similar players

View the comprehensive analytical report

Examine strengths and weaknesses for each player

# Customization
You can customize the tool by:

Modifying the TEAM_METADATA dictionary to update team colors/logos

Adjusting the weights in calculate_similarity() to change similarity calculations

Adding new metrics to the create_comparison_df() function

Changing the visual styling in the CSS sections
