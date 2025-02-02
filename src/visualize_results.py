import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

class BenchmarkVisualizer:
    def __init__(self, results_file: str = "benchmark_evaluations.json"):
        """Initialize visualizer with results file."""
        self.results_file = Path(results_file)
        self.results = self._load_results()
        self.output_dir = Path("figures")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _load_results(self) -> Dict:
        """Load evaluation results from file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def create_model_comparison_plot(self):
        """Create bar plot comparing model performances."""
        scores = self.results["scores"]
        models = ["gpt4", "claude"]
        metrics = ["physical_understanding_avg", "solution_path_avg", "red_herring_avg"]
        
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    "Model": model,
                    "Metric": metric.replace("_avg", "").replace("_", " ").title(),
                    "Score": scores[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(data=df, x="Metric", y="Score", hue="Model")
        chart.set_title("Model Performance Comparison by Metric")
        plt.ylim(0, 2)  # Scores range from 0-2
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png")
        plt.close()
    
    def create_problem_heatmap(self):
        """Create heatmap showing performance across problems."""
        scores = self.results["scores"]
        problems = list(scores["gpt4"]["problem_scores"].keys())
        
        data = {
            "gpt4": [scores["gpt4"]["problem_scores"].get(p, 0) for p in problems],
            "claude": [scores["claude"]["problem_scores"].get(p, 0) for p in problems]
        }
        
        df = pd.DataFrame(data, index=problems)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(df, annot=True, cmap="YlOrRd", vmin=0, vmax=5)
        plt.title("Problem Performance Heatmap")
        plt.tight_layout()
        plt.savefig(self.output_dir / "problem_heatmap.png")
        plt.close()
    
    def create_spider_plot(self):
        """Create spider/radar plot for model capabilities."""
        scores = self.results["scores"]
        metrics = ["physical_understanding_avg", "solution_path_avg", "red_herring_avg"]
        
        # Set up the angles for the radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        # Close the plot by appending first value
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for model in ["gpt4", "claude"]:
            values = [scores[model][metric] for metric in metrics]
            values = np.concatenate((values, [values[0]]))  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_avg", "").replace("_", " ").title() for m in metrics])
        ax.set_ylim(0, 2)
        plt.title("Model Capabilities Comparison")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(self.output_dir / "capabilities_radar.png")
        plt.close()
    
    def create_error_analysis(self):
        """Create visualization of error patterns."""
        evaluations = self.results["evaluations"]
        data = []
        
        for model in ["gpt4", "claude"]:
            for eval_result in evaluations[f"{model}_evaluations"]:
                evaluation = eval_result["evaluation"]
                # Calculate errors
                physical_error = 2 - evaluation["physical_understanding_score"]
                solution_error = 2 - evaluation["solution_path_score"]
                red_herring_error = 1 - evaluation["red_herring_score"]
                
                # Add to data list
                data.extend([
                    {"Model": model, "Component": "Physical Understanding", "Error": physical_error},
                    {"Model": model, "Component": "Solution Path", "Error": solution_error},
                    {"Model": model, "Component": "Red Herring", "Error": red_herring_error}
                ])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Component", y="Error", hue="Model")
        plt.title("Error Distribution by Component")
        plt.ylabel("Error Magnitude")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_analysis.png")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization types."""
        self.create_model_comparison_plot()
        self.create_problem_heatmap()
        self.create_spider_plot()
        self.create_error_analysis()
        print("All visualizations generated in 'figures' directory.")

def create_visualizations():
    """Entry point for visualization generation."""
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all_visualizations()


create_visualizations()