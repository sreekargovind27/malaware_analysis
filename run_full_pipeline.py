#!/usr/bin/env python3
"""
Full Pipeline Runner - Stage 1 + Stage 2

Runs the complete data preparation pipeline:
1. Stage 1: Feasibility analysis (PySpark)
2. Stage 2: Feature engineering (PySpark)

Usage:
    python run_full_pipeline.py [--sample 0.1]
"""
import sys
import time
from datetime import datetime

from config import Config


def run_stage1():
    """Execute Stage 1 feasibility analysis."""
    print("\n" + "=" * 80)
    print("🔍 STAGE 1: FEASIBILITY ANALYSIS (PySpark)")
    print("=" * 80)
    
    try:
        from analysis.stage1.run_stage1 import main as stage1_main
        stage1_main()
        print("\n✅ Stage 1 complete!")
        return True
    except Exception as e:
        print(f"\n❌ Stage 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_stage2():
    """Execute Stage 2 feature engineering."""
    print("\n" + "=" * 80)
    print("🚀 STAGE 2: FEATURE ENGINEERING (PySpark)")
    print("=" * 80)
    
    try:
        from analysis.stage2.run_stage2 import main as stage2_main
        stage2_main()
        print("\n✅ Stage 2 complete!")
        return True
    except Exception as e:
        print(f"\n❌ Stage 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run full pipeline."""
    print("\n" + "=" * 80)
    print("🎯 FULL PIPELINE: STAGE 1 + STAGE 2")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'TEST' if Config.TEST_MODE else 'PRODUCTION'}")
    print(f"Sample: {Config.DATA_SAMPLE_FRACTION * 100:.1f}%")
    print(f"Input: {Config.RAW_DIR_ORIGINAL}")
    print("=" * 80)
    
    overall_start = time.time()
    
    # Ensure directories
    Config.ensure_output_dirs()
    
    # Stage 1
    stage1_success = run_stage1()
    
    if not stage1_success:
        print("\n❌ Pipeline aborted due to Stage 1 failure")
        sys.exit(1)
    
    # Stage 2
    stage2_success = run_stage2()
    
    if not stage2_success:
        print("\n❌ Pipeline aborted due to Stage 2 failure")
        sys.exit(1)
    
    # Summary
    total_time = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("🎉 FULL PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"\n📊 Outputs:")
    print(f"   Stage 1: {Config.STAGE1_FEASIBILITY_DIR}")
    print(f"   Stage 2: {Config.STAGE2_PREPARED_DIR}")
    print(f"\n✅ Ready for Stage 3: Model Training")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full Stage 1 + Stage 2 pipeline')
    parser.add_argument('--sample', type=float, default=None,
                        help='Override sample fraction (0.01-1.0)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.sample is not None:
        if not 0 < args.sample <= 1.0:
            print("❌ Sample must be between 0 and 1.0")
            sys.exit(1)
        Config.DATA_SAMPLE_FRACTION = args.sample
        print(f"⚙️  Overriding sample fraction: {args.sample * 100:.1f}%")
    
    main()
