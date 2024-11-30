import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import numpy as np

class Visualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid")

    def plot_bar(self, results_df: pd.DataFrame, metric: str, unit: str = ""):
        """
        Plot a bar chart for a given metric across all combinations, optionally including a unit.
        """
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Combination', y=metric, data=results_df, palette='viridis')
        plt.title(f'{metric} Across Different Pipelines {f"({unit})" if unit else ""}', fontsize=14)
        plt.ylabel(f'{metric} {f"({unit})" if unit else ""}', fontsize=12)
        plt.xlabel('Pipeline Combinations', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i in ax.containers:
            ax.bar_label(i, fmt='%.3f', padding=3)
            
        plt.tight_layout()
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        # Calculate appropriate major and minor intervals based on data range
        y_range = results_df[metric].max() - results_df[metric].min()
        # Round to a nice number for better readability
        major_interval = np.around(y_range / 8, decimals=2)  # Aim for ~8 major ticks
        # Ensure major interval is not too small
        if major_interval < 0.05:
            major_interval = 0.05
        minor_interval = major_interval / 5  # Five minor ticks between majors
        
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(major_interval))
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(minor_interval))
        plt.show()


    def plot_comparison(self, results_df: pd.DataFrame, metrics: list, time_metrics: list = None, unit: str = ""):
        """
        Plot a line chart comparing multiple metrics across all combinations, 
        optionally including time unit for time metrics.
        """
        if time_metrics:
            metrics_titles = {metric: f"{metric} ({unit})" if metric in time_metrics else metric for metric in metrics}
        else:
            metrics_titles = {metric: metric for metric in metrics}
        
        melted_df = results_df.melt(id_vars=['Combination'], value_vars=metrics, 
                                    var_name='Metric', value_name='Value')
        
        # Update melted DataFrame with custom titles
        melted_df['Metric'] = melted_df['Metric'].map(metrics_titles)
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=melted_df, x='Combination', y='Value', hue='Metric', marker='o')
        plt.title('Metric Comparison Across Pipelines', fontsize=14)
        plt.ylabel('Value', fontsize=12)
        plt.xlabel('Pipeline Combinations', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metrics', fontsize=10)
        plt.tight_layout()
        plt.show()


    def plot_heatmap(self, results_df: pd.DataFrame, metrics: list):
        """
        Plot a heatmap showing the values of different metrics for each pipeline.
        """
        heatmap_data = results_df.set_index('Combination')[metrics]
        plt.figure(figsize=(12, 12))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title('Heatmap of Metrics Across Pipelines', fontsize=14)
        plt.ylabel('Pipeline Combinations', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_clustermap(self, results_df: pd.DataFrame, metrics: list):
        """
        Plot a clustermap showing hierarchically-clustered heatmap of metrics across pipelines.
        """
        clustermap_data = results_df.set_index('Combination')[metrics]
        plt.figure(figsize=(12, 16))
        sns.clustermap(clustermap_data, annot=True, fmt=".2f", cmap='coolwarm', 
                        cbar=True, figsize=(12, 16))
        plt.suptitle('Clustered Heatmap of Metrics Across Pipelines', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()


    def plot_metric_distribution(self, results_df: pd.DataFrame, metric: str):
        """
        Plot the distribution of a single metric across all combinations.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df[metric], kde=True, bins=10, color='blue')
        plt.title(f'Distribution of {metric}', fontsize=14)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_top_n(self, results_df: pd.DataFrame, metric: str, n: int = 5, unit: str = "", ascending: bool = False):
        """
        Plot a bar chart for the top N pipelines based on a specific metric, optionally including a unit.
        """
        # for some reason it shows only n/2 bars if n is not multiplied by 2
        top_n = results_df.nlargest(n*2, metric) if ascending else results_df.nsmallest(n*2, metric)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Combination', y=metric, data=top_n, palette='mako')
        plt.title(f'Top {n} {"Longest" if ascending else "Shortest" } Pipelines by {metric} {f"({unit})" if unit else ""}', fontsize=14)
        plt.ylabel(f'{metric} {f"({unit})" if unit else ""}', fontsize=12)
        plt.xlabel('Pipeline Combinations', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


# Example Usage:
# visualizer = Visualizer()
# visualizer.plot_bar(results_df, 'Accuracy')
# visualizer.plot_comparison(results_df, ['Accuracy', 'F1 Score', 'ROC AUC'])
# visualizer.plot_heatmap(results_df, ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
# visualizer.plot_metric_distribution(results_df, 'F1 Score')
# visualizer.plot_top_n(results_df, 'Accuracy', n=5)
