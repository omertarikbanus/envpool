#!/usr/bin/env python3
"""
Visualize hyperparameter search results from Optuna study.

Usage:
    python3 visualize_hyperopt.py --study-dir hyperopt_studies/ppo_hyperopt_Humanoid-v4_20250713_123456
"""

import argparse
import os
import sys
import json
from pathlib import Path

import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def load_study(study_dir: str) -> optuna.Study:
    """Load Optuna study from pickle file."""
    study_path = os.path.join(study_dir, "study.pkl")
    if not os.path.exists(study_path):
        raise FileNotFoundError(f"Study file not found: {study_path}")
    
    study = optuna.pickle.load_object(study_path)
    return study


def create_optimization_history_plot(study: optuna.Study) -> go.Figure:
    """Create optimization history plot."""
    trials = study.trials
    values = [trial.value for trial in trials if trial.value is not None]
    trial_numbers = [trial.number for trial in trials if trial.value is not None]
    
    # Best value so far
    best_values = []
    best_so_far = float('-inf')
    for value in values:
        if value > best_so_far:
            best_so_far = value
        best_values.append(best_so_far)
    
    fig = go.Figure()
    
    # Individual trial values
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=values,
        mode='markers',
        name='Trial Value',
        opacity=0.6,
        marker=dict(size=6)
    ))
    
    # Best value line
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=best_values,
        mode='lines',
        name='Best Value So Far',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Optimization History',
        xaxis_title='Trial Number',
        yaxis_title='Objective Value (Mean Reward)',
        hovermode='x unified'
    )
    
    return fig


def create_parameter_importance_plot(study: optuna.Study) -> go.Figure:
    """Create parameter importance plot."""
    try:
        importance = optuna.importance.get_param_importances(study)
        
        params = list(importance.keys())
        importances = list(importance.values())
        
        fig = go.Figure(data=[
            go.Bar(x=importances, y=params, orientation='h')
        ])
        
        fig.update_layout(
            title='Parameter Importance',
            xaxis_title='Importance',
            yaxis_title='Parameter',
            height=max(400, len(params) * 30)
        )
        
        return fig
    except Exception as e:
        print(f"Could not create parameter importance plot: {e}")
        return None


def create_parallel_coordinate_plot(study: optuna.Study) -> go.Figure:
    """Create parallel coordinate plot."""
    trials = [trial for trial in study.trials if trial.value is not None]
    
    if len(trials) < 2:
        print("Not enough trials for parallel coordinate plot")
        return None
    
    # Get top 20% of trials
    trials = sorted(trials, key=lambda x: x.value, reverse=True)
    top_trials = trials[:max(1, len(trials) // 5)]
    
    # Prepare data
    params = list(top_trials[0].params.keys())
    dimensions = []
    
    for param in params:
        values = [trial.params[param] for trial in top_trials]
        
        # Handle categorical parameters
        if isinstance(values[0], str) or isinstance(values[0], bool):
            unique_values = list(set(values))
            value_map = {val: i for i, val in enumerate(unique_values)}
            numeric_values = [value_map[val] for val in values]
            
            dimensions.append(dict(
                label=param,
                values=numeric_values,
                tickvals=list(range(len(unique_values))),
                ticktext=unique_values
            ))
        else:
            dimensions.append(dict(
                label=param,
                values=values
            ))
    
    # Add objective values
    dimensions.append(dict(
        label='Objective Value',
        values=[trial.value for trial in top_trials]
    ))
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=[trial.value for trial in top_trials],
                 colorscale='Viridis',
                 showscale=True,
                 colorbar=dict(title="Objective Value")),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title=f'Parallel Coordinate Plot (Top {len(top_trials)} trials)',
        height=600
    )
    
    return fig


def create_hyperparameter_distribution_plots(study: optuna.Study) -> go.Figure:
    """Create distribution plots for each hyperparameter."""
    trials = [trial for trial in study.trials if trial.value is not None]
    
    if len(trials) < 2:
        print("Not enough trials for distribution plots")
        return None
    
    params = list(trials[0].params.keys())
    n_params = len(params)
    
    # Create subplots
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=params,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, param in enumerate(params):
        row = i // cols + 1
        col = i % cols + 1
        
        values = [trial.params[param] for trial in trials]
        
        # Handle different types of parameters
        if isinstance(values[0], (bool, str)):
            # Categorical parameters - use bar plot
            unique_values, counts = zip(*pd.Series(values).value_counts().items())
            fig.add_trace(
                go.Bar(x=list(unique_values), y=list(counts), name=param, showlegend=False),
                row=row, col=col
            )
        else:
            # Numerical parameters - use histogram
            fig.add_trace(
                go.Histogram(x=values, name=param, showlegend=False, nbinsx=20),
                row=row, col=col
            )
    
    fig.update_layout(
        title='Hyperparameter Distributions',
        height=300 * rows,
        showlegend=False
    )
    
    return fig


def generate_report(study_dir: str):
    """Generate a comprehensive HTML report."""
    study = load_study(study_dir)
    
    print(f"Study: {study.study_name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Create plots
    plots = {}
    
    print("Creating optimization history plot...")
    plots['history'] = create_optimization_history_plot(study)
    
    print("Creating parameter importance plot...")
    plots['importance'] = create_parameter_importance_plot(study)
    
    print("Creating parallel coordinate plot...")
    plots['parallel'] = create_parallel_coordinate_plot(study)
    
    print("Creating hyperparameter distribution plots...")
    plots['distributions'] = create_hyperparameter_distribution_plots(study)
    
    # Save plots
    output_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in plots.items():
        if fig is not None:
            html_path = os.path.join(output_dir, f"{name}.html")
            fig.write_html(html_path)
            print(f"Saved {name} plot to: {html_path}")
    
    # Create summary HTML
    create_summary_html(study, output_dir, plots)


def create_summary_html(study: optuna.Study, output_dir: str, plots: dict):
    """Create a summary HTML file with all plots."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hyperparameter Search Results - {study.study_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .plot-container {{ margin: 20px 0; }}
            .best-params {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Hyperparameter Search Results</h1>
            <p><strong>Study:</strong> {study.study_name}</p>
            <p><strong>Number of trials:</strong> {len(study.trials)}</p>
            <p><strong>Best objective value:</strong> {study.best_value:.4f}</p>
        </div>
        
        <div class="best-params">
            <h2>Best Parameters</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    for param, value in study.best_params.items():
        html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    html_content += """
            </table>
        </div>
        
        <h2>Visualizations</h2>
    """
    
    # Add plot iframes
    for name, fig in plots.items():
        if fig is not None:
            html_content += f"""
            <div class="plot-container">
                <h3>{name.replace('_', ' ').title()}</h3>
                <iframe src="{name}.html" width="100%" height="600" frameborder="0"></iframe>
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    summary_path = os.path.join(output_dir, "summary.html")
    with open(summary_path, 'w') as f:
        f.write(html_content)
    
    print(f"Summary report saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize hyperparameter search results.")
    parser.add_argument("--study-dir", type=str, required=True,
                       help="Directory containing the Optuna study")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.study_dir):
        print(f"Error: Study directory not found: {args.study_dir}")
        sys.exit(1)
    
    try:
        generate_report(args.study_dir)
        print(f"\nVisualization complete! Open the summary.html file in your browser:")
        print(f"file://{os.path.abspath(os.path.join(args.study_dir, 'visualizations', 'summary.html'))}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
