"""
Data Quality Analysis - Overall dataset quality checks (PySpark).
"""

from datetime import datetime
from pyspark.sql import functions as F
from .utils import calculate_missing_percentage


def analyze_data_quality(df):
    """
    Analyze overall data quality metrics using PySpark.
    
    Args:
        df: Spark DataFrame (raw data)
        
    Returns:
        dict: Data quality statistics
    """
    print("\n" + "="*70)
    print("ANALYZING DATA QUALITY")
    print("="*70)
    
    total_rows = df.count()
    total_files = df.select('Source_Folder').distinct().count()
    columns = df.columns
    column_count = len(columns)
    
    stats = {
        'total_rows': int(total_rows),
        'total_files': int(total_files),
        'date_analyzed': datetime.now().isoformat(),
        'columns': columns,
        'column_count': column_count,
        'missing_percentage_per_column': {},
        'duplicate_rows': 0,
        'temporal_info': {},
        'file_distribution': {}
    }
    
    # Missing values per column
    print("\n  Checking missing values per column...")
    for col in columns:
        missing_pct = calculate_missing_percentage(df, col)
        stats['missing_percentage_per_column'][col] = missing_pct
    
    # Overall missing percentage
    total_cells = total_rows * column_count
    if total_cells > 0:
        # Count nulls across all columns
        null_counts = [df.filter(F.col(col).isNull()).count() for col in columns]
        total_missing = sum(null_counts)
        stats['overall_missing_percentage'] = round((total_missing / total_cells) * 100, 2)
    else:
        stats['overall_missing_percentage'] = 0.0
    
    # Duplicate rows
    print("  Checking for duplicate rows...")
    stats['duplicate_rows'] = int(total_rows - df.dropDuplicates().count())
    
    # Temporal analysis
    print("  Analyzing temporal distribution...")
    if 'ts' in df.columns:
        # Cast to double for timestamp operations
        df_temp = df.withColumn('ts_numeric', F.col('ts').cast('double'))
        
        ts_stats = df_temp.agg(
            F.min('ts_numeric').alias('min_ts'),
            F.max('ts_numeric').alias('max_ts')
        ).collect()[0]
        
        if ts_stats['min_ts'] is not None and ts_stats['max_ts'] is not None:
            min_ts = float(ts_stats['min_ts'])
            max_ts = float(ts_stats['max_ts'])
            duration_days = (max_ts - min_ts) / 86400  # seconds to days
            
            stats['temporal_info'] = {
                'min_timestamp': min_ts,
                'max_timestamp': max_ts,
                'duration_days': round(duration_days, 2),
                'min_date': datetime.fromtimestamp(min_ts).isoformat(),
                'max_date': datetime.fromtimestamp(max_ts).isoformat()
            }
    
    # File distribution
    print("  Calculating file distribution...")
    file_dist = df.groupBy('Source_Folder').count().collect()
    stats['file_distribution'] = {row['Source_Folder']: int(row['count']) for row in file_dist}
    
    # Summary
    print(f"\n✓ Total rows: {stats['total_rows']:,}")
    print(f"✓ Total files: {stats['total_files']}")
    print(f"✓ Total columns: {stats['column_count']}")
    print(f"✓ Overall missing: {stats['overall_missing_percentage']}%")
    print(f"✓ Duplicate rows: {stats['duplicate_rows']:,}")
    
    if stats['temporal_info']:
        print(f"✓ Data spans: {stats['temporal_info']['duration_days']:.1f} days")
    
    return stats
