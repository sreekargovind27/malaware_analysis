"""
Diagnostic script to identify why Stage 1 is returning wrong results.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark
spark = SparkSession.builder \
    .appName("DiagnoseIssues") \
    .master("local[*]") \
    .getOrCreate()

# Load your test CSV (adjust path as needed)
df = spark.read.csv(
    "data/raw_test/*.csv",  # or wherever your test data is
    header=True,
    inferSchema=True
)

print("="*70)
print("DIAGNOSTIC REPORT")
print("="*70)

# 1. Check columns
print("\n1. COLUMNS:")
print(f"   Total columns: {len(df.columns)}")
print(f"   Column names: {df.columns}")
print(f"   Has 'label': {'label' in df.columns}")

# 2. Check detailed-label values
print("\n2. LABEL COLUMN:")
if 'label' in df.columns:
    print("   Sample values:")
    df.select('label').show(10, truncate=False)
    
    # Check for malicious
    malicious_count = df.filter(
        F.lower(F.col('label')).contains('malicious')
    ).count()
    print(f"   Rows containing 'malicious': {malicious_count}")
    
    # Check for benign
    benign_count = df.filter(
        F.lower(F.col('label')).contains('benign')
    ).count()
    print(f"   Rows containing 'benign': {benign_count}")
    
    # Check for NULL
    null_count = df.filter(F.col('label').isNull()).count()
    print(f"   NULL values: {null_count}")
else:
    print("   ❌ Column 'detailed-label' NOT FOUND!")

# 3. Check data types
print("\n3. DATA TYPES (checking numeric detection bug):")
for field in df.schema.fields[:10]:  # First 10 fields
    print(f"   {field.name}:")
    print(f"      - dataType: {field.dataType}")
    print(f"      - str(dataType): '{str(field.dataType)}'")
    print(f"      - typeName(): '{field.dataType.typeName()}'")
    
    # Old buggy check
    old_check = str(field.dataType) in ['IntegerType', 'LongType', 'FloatType', 'DoubleType']
    # New correct check
    new_check = field.dataType.typeName() in ['integer', 'long', 'float', 'double']
    
    print(f"      - Old check (BUGGY): {old_check}")
    print(f"      - New check (FIXED): {new_check}")
    print()

# 4. Count numeric features (both ways)
print("\n4. NUMERIC FEATURE COUNT:")

# OLD BUGGY WAY
numeric_cols_old = [field.name for field in df.schema.fields 
                    if str(field.dataType) in ['IntegerType', 'LongType', 'FloatType', 'DoubleType']]
print(f"   Old method (BUGGY): {len(numeric_cols_old)} numeric columns")

# NEW FIXED WAY
numeric_cols_new = [field.name for field in df.schema.fields 
                    if field.dataType.typeName() in ['integer', 'long', 'float', 'double']]
print(f"   New method (FIXED): {len(numeric_cols_new)} numeric columns")
print(f"   Numeric columns: {numeric_cols_new}")

# 5. Total row count
print(f"\n5. TOTAL ROWS: {df.count():,}")

# 6. Check Source_Folder
print("\n6. SOURCE_FOLDER:")
if 'Source_Folder' in df.columns:
    source_folders = df.select('Source_Folder').distinct().collect()
    print(f"   Unique Source_Folders: {len(source_folders)}")
    for row in source_folders[:5]:
        print(f"      - {row['Source_Folder']}")
else:
    print("   ❌ Column 'Source_Folder' NOT FOUND!")

print("="*70)

spark.stop()
