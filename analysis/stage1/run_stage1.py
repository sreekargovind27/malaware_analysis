#!/usr/bin/env python3
"""
Stage 1: Raw Data Feasibility Analysis - Main Runner
Works on both local and Databricks.

Runs all feasibility checks on raw IoT-23 data and generates
comprehensive reports for each model type.

Analysis modules:
1. Data quality checks
2. Binary classification feasibility
3. Multiclass classification feasibility
4. Malware family feasibility
5. Unsupervised models (autoencoder, clustering, GAN)
6. Graph structure analysis

Generates JSON reports for each module plus summary.
"""

import os
import sys
from datetime import datetime

from config import Config
from analysis.stage1.utils import load_raw_data, save_json_report
from analysis.stage1.data_quality import analyze_data_quality
from analysis.stage1.binary_multiclass import (
    analyze_binary_classification,
    analyze_multiclass_classification
)
from analysis.stage1.malware_family import analyze_malware_family_classification
from analysis.stage1.graph_analysis import analyze_graph_structure
from analysis.stage1.unsupervised_models import (
    analyze_autoencoder_feasibility,
    analyze_clustering_feasibility,
    analyze_gan_feasibility_for_all_tasks
)


def generate_summary(all_stats: dict) -> dict:
    """
    Generate overall summary with Go/No-Go recommendations.
    
    Args:
        all_stats: Dictionary of all analysis results
        
    Returns:
        dict: Summary with recommendations
    """
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'environment': Config.get_environment(),
        'data_source': Config.RAW_DIR_ORIGINAL,
        'total_rows': all_stats.get('data_quality', {}).get('total_rows', 0),
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
        summary['overall_message'] = (
            f'{nogo_count} models are not feasible. Review individual reports.'
        )
    elif warning_count > len(models) / 2:
        summary['overall_recommendation'] = 'PROCEED WITH CAUTION'
        summary['overall_message'] = (
            f'{warning_count} models have warnings. '
            f'Consider data augmentation or filtering.'
        )
    else:
        summary['overall_recommendation'] = 'PROCEED'
        summary['overall_message'] = (
            f'{go_count} models are feasible. Ready for feature engineering.'
        )

    return summary


def main():
    """Main execution function."""

    Config.ensure_output_dirs()

    print("\n" + "=" * 70)
    print("STAGE 1: RAW DATA FEASIBILITY ANALYSIS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Environment: {Config.get_environment()}")
    print(f"Data source: {Config.RAW_DIR_ORIGINAL}")
    print(f"Output directory: {Config.STAGE1_FEASIBILITY_DIR}")
    print("=" * 70)

    overall_start = datetime.now()

    # Initialize Spark
    print("\nInitializing Spark session...")
    spark = Config.get_spark_session("Stage1-Feasibility-Analysis")
    print(f"✅ Spark version: {spark.version}")

    # Load raw data once
    try:
        df = load_raw_data(spark)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        if not Config.is_databricks():
            spark.stop()
        return

    # Dictionary to store all results
    all_stats = {}

    # =========================================================================
    # 1. DATA QUALITY ANALYSIS
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS [1/6]: DATA QUALITY")
    print("=" * 70)

    try:
        all_stats['data_quality'] = analyze_data_quality(df)
        save_json_report(
            all_stats['data_quality'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'data_quality.json')
        )
        print("✅ Data quality analysis complete")
    except Exception as e:
        print(f"\n❌ Error in data quality analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 2. BINARY CLASSIFICATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS [2/6]: BINARY CLASSIFICATION")
    print("=" * 70)

    try:
        all_stats['binary_classification'] = analyze_binary_classification(df)
        save_json_report(
            all_stats['binary_classification'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'binary_feasibility.json')
        )
        print("✅ Binary classification analysis complete")
    except Exception as e:
        print(f"\n❌ Error in binary classification analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 3. MULTICLASS CLASSIFICATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS [3/6]: MULTICLASS CLASSIFICATION")
    print("=" * 70)

    try:
        all_stats['multiclass_classification'] = analyze_multiclass_classification(df)
        save_json_report(
            all_stats['multiclass_classification'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'multiclass_feasibility.json')
        )
        print("✅ Multiclass classification analysis complete")
    except Exception as e:
        print(f"\n❌ Error in multiclass classification analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 4. MALWARE FAMILY CLASSIFICATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS [4/6]: MALWARE FAMILY CLASSIFICATION")
    print("=" * 70)

    try:
        all_stats['malware_family_classification'] = analyze_malware_family_classification(df)
        save_json_report(
            all_stats['malware_family_classification'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'malware_family_feasibility.json')
        )
        print("✅ Malware family analysis complete")
    except Exception as e:
        print(f"\n❌ Error in malware family analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 5. UNSUPERVISED MODELS
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS [5/6]: UNSUPERVISED MODELS")
    print("=" * 70)

    try:
        # Autoencoder
        all_stats['autoencoder_feasibility'] = analyze_autoencoder_feasibility(df)
        save_json_report(
            all_stats['autoencoder_feasibility'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'autoencoder_feasibility.json')
        )

        # Clustering
        all_stats['clustering_feasibility'] = analyze_clustering_feasibility(df)
        save_json_report(
            all_stats['clustering_feasibility'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'clustering_feasibility.json')
        )

        # GAN
        all_stats['gan_candidates'] = analyze_gan_feasibility_for_all_tasks(df)
        save_json_report(
            all_stats['gan_candidates'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'gan_feasibility.json')
        )

        print("✅ Unsupervised models analysis complete")
    except Exception as e:
        print(f"\n❌ Error in unsupervised models analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 6. GRAPH STRUCTURE ANALYSIS
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS [6/6]: GRAPH STRUCTURE (GNN)")
    print("=" * 70)

    try:
        all_stats['graph_feasibility'] = analyze_graph_structure(df)
        save_json_report(
            all_stats['graph_feasibility'],
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'graph_feasibility.json')
        )
        print("✅ Graph structure analysis complete")
    except Exception as e:
        print(f"\n❌ Error in graph structure analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 7. GENERATE SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)

    try:
        summary = generate_summary(all_stats)
        save_json_report(
            summary,
            os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'feasibility_summary.json')
        )
        print("✅ Summary report generated")
    except Exception as e:
        print(f"\n❌ Error generating summary: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    total_time = (datetime.now() - overall_start).total_seconds()

    print("\n" + "=" * 70)
    print("🎉 STAGE 1 COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"🌍 Environment: {Config.get_environment()}")
    print(f"📁 Reports saved to: {Config.STAGE1_FEASIBILITY_DIR}")

    print("\n📊 Feasibility Summary:")
    if 'model_feasibility' in summary:
        for model, info in summary['model_feasibility'].items():
            status_emoji = {
                'GO': '✅',
                'WARNING': '⚠️',
                'NO-GO': '❌',
                'RECOMMENDED': '💡',
                'NOT_APPLICABLE': '➖'
            }.get(info['status'], '❓')

            print(f"   {status_emoji} {model}: {info['status']}")

    print(f"\n🎯 Overall Recommendation: {summary.get('overall_recommendation', 'N/A')}")
    print(f"   {summary.get('overall_message', 'N/A')}")

    print("\n📄 Reports generated:")
    print(f"   • data_quality.json")
    print(f"   • binary_feasibility.json")
    print(f"   • multiclass_feasibility.json")
    print(f"   • malware_family_feasibility.json")
    print(f"   • autoencoder_feasibility.json")
    print(f"   • clustering_feasibility.json")
    print(f"   • gan_feasibility.json")
    print(f"   • graph_feasibility.json")
    print(f"   • feasibility_summary.json")

    print("\n🚀 Next Steps:")
    print("   1. Review feasibility reports in outputs/stage1_feasibility/")
    print("   2. If overall recommendation is PROCEED, run Stage 2 (feature engineering)")
    print("   3. Use: python analysis/stage2/run_stage2.py")

    print("=" * 70)

    # Only stop Spark on local (Databricks manages cluster)
    if not Config.is_databricks():
        spark.stop()
        print("🧹 Spark session closed.")


if __name__ == "__main__":
    main()
