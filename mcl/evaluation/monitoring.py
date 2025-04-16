import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple, Any, Optional, Union
import os
import json
from datetime import datetime

class PerformanceMonitor:
    """
    System for tracking and visualizing model performance over time
    """
    def __init__(
        self,
        log_dir: str = 'performance_logs',
        model_name: str = 'voicevault',
        demographics_columns: List[str] = None
    ):
        self.log_dir = log_dir
        self.model_name = model_name
        self.demographics_columns = demographics_columns or []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_log = pd.DataFrame()
        self.confusion_matrices = {}
        self.roc_curves = {}
        
        # Performance by demographics
        self.demographic_metrics = {}
        
        # Embedding visualizations
        self.embedding_logs = {}
        
        # Initialize shadow model metrics
        self.shadow_model_metrics = {}
        
    def log_evaluation_results(
        self,
        metrics: Dict[str, float],
        subset_name: str = 'validation',
        demographics: pd.DataFrame = None,
        timestamp: float = None
    ):
        """
        Log evaluation metrics
        
        Args:
            metrics: Dictionary of metrics (e.g., accuracy, EER)
            subset_name: Name of the evaluation subset
            demographics: DataFrame with demographic information
            timestamp: Time of evaluation (default: current time)
        """
        timestamp = timestamp or time.time()
        human_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'human_time': human_time,
            'model_name': self.model_name,
            'subset': subset_name,
            **metrics
        }
        
        # Add to metrics log
        self.metrics_log = pd.concat([self.metrics_log, pd.DataFrame([log_entry])], ignore_index=True)
        
        # Log demographics-based metrics if available
        if demographics is not None and not demographics.empty:
            self._log_demographic_metrics(metrics, demographics, subset_name, timestamp)
        
        # Save to disk
        self._save_metrics_log()
    
    def _log_demographic_metrics(
        self,
        metrics: Dict[str, float],
        demographics: pd.DataFrame,
        subset_name: str,
        timestamp: float
    ):
        """Log metrics stratified by demographic factors"""
        for column in self.demographics_columns:
            if column in demographics.columns:
                # Create a key for this demographic attribute
                demo_key = f"{subset_name}_{column}"
                
                # Initialize if this demographic hasn't been seen before
                if demo_key not in self.demographic_metrics:
                    self.demographic_metrics[demo_key] = []
                
                # Group metrics by demographic category
                for category, group in demographics.groupby(column):
                    category_metrics = {
                        'timestamp': timestamp,
                        'human_time': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        'model_name': self.model_name,
                        'subset': subset_name,
                        'demographic': column,
                        'category': category,
                        'count': len(group),
                        **metrics
                    }
                    
                    self.demographic_metrics[demo_key].append(category_metrics)
        
        # Save to disk
        self._save_demographic_metrics()
    
    def log_confusion_matrix(
        self,
        y_true: List,
        y_pred: List,
        subset_name: str = 'validation',
        class_names: List[str] = None,
        timestamp: float = None
    ):
        """
        Log confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            subset_name: Name of the evaluation subset
            class_names: Names of the classes
            timestamp: Time of evaluation (default: current time)
        """
        timestamp = timestamp or time.time()
        human_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'human_time': human_time,
            'subset': subset_name,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        # Add to confusion matrix log
        cm_key = f"{subset_name}_{human_time}"
        self.confusion_matrices[cm_key] = log_entry
        
        # Save to disk
        self._save_confusion_matrices()
    
    def log_roc_curve(
        self,
        y_true: List,
        y_score: List,
        subset_name: str = 'validation',
        timestamp: float = None
    ):
        """
        Log ROC curve data
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            subset_name: Name of the evaluation subset
            timestamp: Time of evaluation (default: current time)
        """
        timestamp = timestamp or time.time()
        human_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'human_time': human_time,
            'subset': subset_name,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # Add to ROC curve log
        roc_key = f"{subset_name}_{human_time}"
        self.roc_curves[roc_key] = log_entry
        
        # Save to disk
        self._save_roc_curves()
    
    def log_embeddings(
        self,
        embeddings: np.ndarray,
        labels: List,
        subset_name: str = 'validation',
        metadata: Dict = None,
        timestamp: float = None
    ):
        """
        Log embeddings for visualization
        
        Args:
            embeddings: Embedding vectors
            labels: Labels for each embedding
            subset_name: Name of the evaluation subset
            metadata: Additional metadata for embeddings
            timestamp: Time of evaluation (default: current time)
        """
        timestamp = timestamp or time.time()
        human_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'human_time': human_time,
            'subset': subset_name,
            'embeddings_shape': embeddings.shape,
            'labels': labels,
            'metadata': metadata or {}
        }
        
        # Save embeddings separately due to size
        embeddings_dir = os.path.join(self.log_dir, 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        
        embeddings_path = os.path.join(embeddings_dir, 
                                      f"{self.model_name}_{subset_name}_{human_time}.npy")
        np.save(embeddings_path, embeddings)
        log_entry['embeddings_path'] = embeddings_path
        
        # Add to embeddings log
        embedding_key = f"{subset_name}_{human_time}"
        self.embedding_logs[embedding_key] = log_entry
        
        # Save to disk
        self._save_embedding_logs()
    
    def log_shadow_model_metrics(
        self,
        shadow_model_name: str,
        metrics: Dict[str, float],
        subset_name: str = 'validation',
        timestamp: float = None
    ):
        """
        Log shadow model metrics for drift detection
        
        Args:
            shadow_model_name: Name of the shadow model
            metrics: Dictionary of metrics
            subset_name: Name of the evaluation subset
            timestamp: Time of evaluation (default: current time)
        """
        timestamp = timestamp or time.time()
        human_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create log entry
        # evaluation/monitoring.py (continued)
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'human_time': human_time,
            'model_name': shadow_model_name,
            'subset': subset_name,
            **metrics
        }
        
        # Initialize for this shadow model if needed
        if shadow_model_name not in self.shadow_model_metrics:
            self.shadow_model_metrics[shadow_model_name] = []
        
        # Add to shadow model log
        self.shadow_model_metrics[shadow_model_name].append(log_entry)
        
        # Save to disk
        self._save_shadow_model_metrics()
    
    def detect_performance_drift(
        self,
        metric_name: str = 'accuracy',
        window_size: int = 5,
        threshold: float = 0.05
    ) -> bool:
        """
        Detect drift in model performance
        
        Args:
            metric_name: Metric to monitor for drift
            window_size: Size of the sliding window
            threshold: Threshold for detecting drift
            
        Returns:
            True if drift detected, False otherwise
        """
        if len(self.metrics_log) < window_size + 1:
            return False
        
        # Calculate moving average
        recent_metrics = self.metrics_log.sort_values('timestamp', ascending=False)
        recent_values = recent_metrics[metric_name].values[:window_size+1]
        
        # Compare current value to moving average of previous window
        current = recent_values[0]
        prev_window_avg = np.mean(recent_values[1:window_size+1])
        
        # Detect significant deviation
        drift_detected = abs(current - prev_window_avg) > threshold
        
        return drift_detected
    
    def compare_with_shadow_models(
        self,
        metric_name: str = 'accuracy',
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Compare current model with shadow models to detect drift
        
        Args:
            metric_name: Metric to compare
            threshold: Threshold for detecting significant difference
            
        Returns:
            Dictionary with drift detection results
        """
        results = {}
        
        # Get current model's latest metric
        if len(self.metrics_log) == 0:
            return results
        
        current_metrics = self.metrics_log.sort_values('timestamp', ascending=False).iloc[0]
        current_value = current_metrics[metric_name]
        
        # Compare with each shadow model
        for shadow_name, shadow_logs in self.shadow_model_metrics.items():
            if not shadow_logs:
                continue
            
            # Get latest shadow model metric
            shadow_value = shadow_logs[-1][metric_name]
            
            # Calculate difference and detect drift
            difference = abs(current_value - shadow_value)
            drift_detected = difference > threshold
            
            results[shadow_name] = {
                'current_value': current_value,
                'shadow_value': shadow_value,
                'difference': difference,
                'drift_detected': drift_detected
            }
        
        return results
    
    def visualize_metrics_over_time(
        self,
        metric_names: List[str] = None,
        subset_name: str = 'validation',
        save_path: str = None
    ):
        """
        Visualize metrics over time
        
        Args:
            metric_names: List of metrics to visualize (default: all numeric)
            subset_name: Name of the evaluation subset
            save_path: Path to save visualization
        """
        if len(self.metrics_log) == 0:
            print("No metrics logged yet.")
            return
        
        # Filter for subset
        subset_metrics = self.metrics_log[self.metrics_log['subset'] == subset_name]
        
        if len(subset_metrics) == 0:
            print(f"No metrics for subset '{subset_name}'.")
            return
        
        # Determine metrics to plot
        if metric_names is None:
            numeric_cols = subset_metrics.select_dtypes(include=[np.number]).columns
            metric_names = [col for col in numeric_cols 
                          if col not in ['timestamp']]
        
        # Create figure
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4*len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metric_names):
            if metric in subset_metrics:
                axes[i].plot(subset_metrics['human_time'], subset_metrics[metric])
                axes[i].set_title(f"{metric} over time")
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
                # Rotate x labels for readability
                plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def visualize_demographic_metrics(
        self,
        demographic: str,
        metric_name: str = 'accuracy',
        subset_name: str = 'validation',
        save_path: str = None
    ):
        """
        Visualize metrics across demographic groups
        
        Args:
            demographic: Demographic attribute to analyze
            metric_name: Metric to visualize
            subset_name: Name of the evaluation subset
            save_path: Path to save visualization
        """
        demo_key = f"{subset_name}_{demographic}"
        
        if demo_key not in self.demographic_metrics or not self.demographic_metrics[demo_key]:
            print(f"No metrics for demographic '{demographic}' in subset '{subset_name}'.")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.demographic_metrics[demo_key])
        
        # Get the latest evaluation for each category
        latest_timestamp = df['timestamp'].max()
        latest_df = df[df['timestamp'] == latest_timestamp]
        
        # Plot bar chart of metric by category
        plt.figure(figsize=(10, 6))
        plt.bar(latest_df['category'], latest_df[metric_name])
        plt.title(f"{metric_name} by {demographic}")
        plt.xlabel(demographic)
        plt.ylabel(metric_name)
        plt.grid(axis='y')
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def visualize_embedding_space(
        self,
        embedding_key: str = None,
        save_path: str = None
    ):
        """
        Visualize embedding space using t-SNE
        
        Args:
            embedding_key: Key for specific embedding log (default: latest)
            save_path: Path to save visualization
        """
        # Get embedding log
        if embedding_key is None:
            if not self.embedding_logs:
                print("No embeddings logged yet.")
                return
            # Use the latest
            embedding_key = sorted(self.embedding_logs.keys(), 
                                 key=lambda k: self.embedding_logs[k]['timestamp'])[-1]
        
        if embedding_key not in self.embedding_logs:
            print(f"No embedding log found for key '{embedding_key}'.")
            return
        
        embedding_log = self.embedding_logs[embedding_key]
        
        # Load embeddings
        embeddings = np.load(embedding_log['embeddings_path'])
        labels = embedding_log['labels']
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Convert labels to integers if they're not already
        unique_labels = sorted(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        label_ids = [label_to_id[label] for label in labels]
        
        # Create scatter plot
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=label_ids, cmap='tab10', alpha=0.7)
        
        # Add legend
        legend1 = plt.legend(scatter.legend_elements()[0], unique_labels, 
                          title="Classes", loc="best")
        plt.gca().add_artist(legend1)
        
        plt.title(f"Embedding Space Visualization ({embedding_log['subset']})")
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def _save_metrics_log(self):
        """Save metrics log to disk"""
        metrics_path = os.path.join(self.log_dir, f"{self.model_name}_metrics_log.csv")
        self.metrics_log.to_csv(metrics_path, index=False)
    
    def _save_demographic_metrics(self):
        """Save demographic metrics to disk"""
        for demo_key, metrics in self.demographic_metrics.items():
            demo_path = os.path.join(self.log_dir, 
                                   f"{self.model_name}_{demo_key}_demographics.json")
            with open(demo_path, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def _save_confusion_matrices(self):
        """Save confusion matrices to disk"""
        cm_path = os.path.join(self.log_dir, 
                             f"{self.model_name}_confusion_matrices.json")
        with open(cm_path, 'w') as f:
            json.dump(self.confusion_matrices, f, indent=2)
    
    def _save_roc_curves(self):
        """Save ROC curves to disk"""
        roc_path = os.path.join(self.log_dir, 
                              f"{self.model_name}_roc_curves.json")
        with open(roc_path, 'w') as f:
            json.dump(self.roc_curves, f, indent=2)
    
    def _save_embedding_logs(self):
        """Save embedding logs to disk"""
        log_path = os.path.join(self.log_dir, 
                              f"{self.model_name}_embedding_logs.json")
        # Remove actual embeddings from log before saving (they're saved separately)
        logs_to_save = {}
        for key, log in self.embedding_logs.items():
            logs_to_save[key] = {k: v for k, v in log.items() if k != 'embeddings'}
        
        with open(log_path, 'w') as f:
            json.dump(logs_to_save, f, indent=2)
    
    def _save_shadow_model_metrics(self):
        """Save shadow model metrics to disk"""
        for shadow_name, metrics in self.shadow_model_metrics.items():
            shadow_path = os.path.join(self.log_dir, 
                                     f"{shadow_name}_shadow_metrics.json")
            with open(shadow_path, 'w') as f:
                json.dump(metrics, f, indent=2)

