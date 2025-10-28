"""
Stage 1: Raw Data Feasibility Analysis - Main Runner

This script runs all feasibility checks on raw IoT-23 data and generates
comprehensive reports for each model type.

Works both locally and on Databricks (PySpark).

Usage:
    Local: python analysis/stage1/run_stage1.py
    Databricks: Run as notebook or submit as job
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from datetime import datetime
from config import Config
from analysis.stage1.utils import load_raw_data, save_json_report
from analysis.stage1.data_quality import analyze_data_quality
from analysis.stage1.binary_multiclass import analyze_binary_classification, analyze_multiclass_classification
from analysis.stage1.malware_family import analyze_malware_family_classification
from analysis.stage1.graph_analysis import analyze_graph_structure
from analysis.stage1.unsupervised_models import (
    analyze_autoencoder_feasibility,
    analyze_clustering_feasibility,
    analyze_gan_feasibility_for_all_tasks
)


def generate_summary(all_stats):
    """
    Generate overall summary with Go/No-Go recommendations.
    
    Args:
        all_stats: Dictionary of all analysis results
        
    Returns:
        dict: Summary with recommendations
    """
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'data_source': Config.RAW_DIR_ORIGINAL,
        'total_rows': all_stats['data_quality']['total_rows'],
        'model_feasibility': {}
    }
    
    # Extract feasibility from each model
    models = {
        'binary_classification': all_stats.get('binary_classification', {}),
        'multiclass_classification': all_stats.get('multiclass_classification', {}),
        'malware_family_classification': all_stats.get('malware_family_classification', {}),
        'autoencoder_anomaly_detection': all_stats.get('autoencoder_feasibility', {}),
        'kmeans_clustering': all_stats.get('clustering_feasibility', {}),
        'gan_augmentation': all_stats.get('gan_candidates', {}),
        'gnn_graph_classification': all_stats.get('graph_feasibility', {})
    }
    
    for model_name, stats in models.items():
        if stats and 'feasibility' in stats:
            summary['model_feasibility'][model_name] = {
                'status': stats['feasibility'],
                'reason': stats.get('reason', 'N/A')
            }
    
    # Overall recommendation
    go_count = sum(1 for s in summary['model_feasibility'].values() if s['status'] == 'GO')
    warning_count = sum(1 for s in summary['model_feasibility'].values() if s['status'] == 'WARNING')
    nogo_count = sum(1 for s in summary['model_feasibility'].values() if s['status'] == 'NO-GO')
    
    if nogo_count > 0:
        summary['overall_recommendation'] = 'CAUTION'
        summary['overall_message'] = f'{nogo_count} models are not feasible. Review individual reports.'
    elif warning_count > len(models) / 2:
        summary['overall_recommendation'] = 'PROCEED WITH CAUTION'
        summary['overall_message'] = f'{warning_count} models have warnings. Consider data augmentation or filtering.'
    else:
        summary['overall_recommendation'] = 'PROCEED'
        summary['overall_message'] = f'{go_count} models are feasible. Ready for feature engineering.'
    
    return summary


def main():
    """Main execution function."""

    Config.ensure_output_dirs()  # ✅ ADD THIS LINE

    print("\n" + "="*70)
    print("STAGE 1: RAW DATA FEASIBILITY ANALYSIS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data source: {Config.RAW_DIR_ORIGINAL}")
    print(f"Output directory: {Config.STAGE1_FEASIBILITY_DIR}")
    print("="*70)
    
    overall_start = datetime.now()
    
    # Initialize Spark
    print("\nInitializing Spark session...")
    spark = Config.get_spark_session("Stage1-Feasibility-Analysis")
    print(f"✓ Spark version: {spark.version}")
    
    # Load raw data
    try:
        df = load_raw_data(spark)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return
    
    # Dictionary to store all results
    all_stats = {}
    
    # 1. Data Quality Analysis
    try:
        all_stats['data_quality'] = analyze_data_quality(df)
        save_json_report(
            all_stats['data_quality'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'data_quality.json')
        )
    except Exception as e:
        print(f"\n❌ Error in data quality analysis: {e}")
    
    # 2. Binary Classification
    try:
        all_stats['binary_classification'] = analyze_binary_classification(df)
        save_json_report(
            all_stats['binary_classification'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'binary_feasibility.json')
        )
    except Exception as e:
        print(f"\n❌ Error in binary classification analysis: {e}")
    
    # 3. Multi-Class Classification
    try:
        all_stats['multiclass_classification'] = analyze_multiclass_classification(df)
        save_json_report(
            all_stats['multiclass_classification'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'multiclass_feasibility.json')
        )
    except Exception as e:
        print(f"\n❌ Error in multiclass classification analysis: {e}")
    
    # 4. Malware Family Classification
    try:
        all_stats['malware_family_classification'] = analyze_malware_family_classification(df)
        save_json_report(
            all_stats['malware_family_classification'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'malware_family_feasibility.json')
        )
    except Exception as e:
        print(f"\n❌ Error in malware family analysis: {e}")
    
    # 5. Graph Structure (GNN)
    try:
        all_stats['graph_feasibility'] = analyze_graph_structure(df)
        save_json_report(
            all_stats['graph_feasibility'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'graph_feasibility.json')
        )
    except Exception as e:
        print(f"\n❌ Error in graph structure analysis: {e}")
    
    # 6. Autoencoder
    try:
        all_stats['autoencoder_feasibility'] = analyze_autoencoder_feasibility(df)
        save_json_report(
            all_stats['autoencoder_feasibility'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'autoencoder_feasibility.json')
        )
    except Exception as e:
        print(f"\n❌ Error in autoencoder analysis: {e}")
    
    # 7. K-Means Clustering
    try:
        all_stats['clustering_feasibility'] = analyze_clustering_feasibility(df)
        save_json_report(
            all_stats['clustering_feasibility'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'clustering_feasibility.json')
        )
    except Exception as e:
        print(f"\n❌ Error in clustering analysis: {e}")
    
    # 8. GAN Candidates
    try:
        all_stats['gan_candidates'] = analyze_gan_feasibility_for_all_tasks(df)
        save_json_report(
            all_stats['gan_candidates'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'gan_candidates.json')
        )
    except Exception as e:
        print(f"\n❌ Error in GAN candidates analysis: {e}")
    
    # 9. Generate Summary
    try:
        summary = generate_summary(all_stats)
        save_json_report(
            summary,
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'summary.json')
        )
    except Exception as e:
        print(f"\n❌ Error generating summary: {e}")
    
    # Final report
    overall_end = datetime.now()
    duration = (overall_end - overall_start).total_seconds()
    
    print("\n" + "="*70)
    print("STAGE 1 ANALYSIS COMPLETE")
    print("="*70)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Output directory: {Config.STAGE1_FEASIBILITY_DIR}")
    print("\nGenerated reports:")
    print("  ✓ summary.json")
    print("  ✓ data_quality.json")
    print("  ✓ binary_feasibility.json")
    print("  ✓ multiclass_feasibility.json")
    print("  ✓ malware_family_feasibility.json")
    print("  ✓ graph_feasibility.json")
    print("  ✓ autoencoder_feasibility.json")
    print("  ✓ clustering_feasibility.json")
    print("  ✓ gan_candidates.json")
    
    if 'summary' in locals():
        print("\n" + "="*70)
        print("OVERALL RECOMMENDATION")
        print("="*70)
        print(f"Status: {summary['overall_recommendation']}")
        print(f"Message: {summary['overall_message']}")
        print("\nModel-by-model status:")
        for model, status_info in summary['model_feasibility'].items():
            status_emoji = "✅" if status_info['status'] == 'GO' else "⚠️" if status_info['status'] == 'WARNING' else "❌"
            print(f"  {status_emoji} {model}: {status_info['status']}")
    
    print("="*70)
    
    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()
