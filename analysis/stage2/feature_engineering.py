# In analysis/stage2/feature_engineering.py

import glob
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config

CHUNK_SIZE = 1_000_000
MAX_WORKERS = min(16, cpu_count())


def scan_single_file(file_path):
    """Scans a single file to extract unique values from categorical columns."""
    proto_values, conn_state_values = set(), set()
    try:
        df_sample = pd.read_csv(file_path, usecols=['proto', 'conn_state'], low_memory=False, on_bad_lines='skip',
                                comment='#')
        if 'proto' in df_sample.columns:
            proto_values.update(
                df_sample['proto'].fillna('unknown').astype(str).replace(['-', '', 'nan', 'None'], 'unknown'))
        if 'conn_state' in df_sample.columns:
            conn_state_values.update(
                df_sample['conn_state'].fillna('unknown').astype(str).replace(['-', '', 'nan', 'None'], 'unknown'))
    except Exception as e:
        tqdm.write(f"  Warning: Could not scan {os.path.basename(file_path)}: {e}")
    return proto_values, conn_state_values


def scan_categorical_values(csv_files):
    """Scans all CSV files in parallel to build a schema of all possible categorical values."""
    print("\n" + "=" * 70)
    print("🔍 SCANNING FILES FOR CATEGORICAL VALUES (Parallel)")
    print("=" * 70)
    all_proto_values, all_conn_state_values = set(), set()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scan_single_file, fp): fp for fp in csv_files}
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Scanning files"):
            proto_vals, conn_vals = future.result()
            all_proto_values.update(proto_vals)
            all_conn_state_values.update(conn_vals)
    print(f"\n✅ Schema discovery complete.")
    return {'proto': sorted(all_proto_values), 'conn_state': sorted(all_conn_state_values)}


def process_single_csv(args):
    """Wrapper function for parallel processing."""
    file_path, all_categorical_values = args
    file_key = os.path.basename(file_path).replace('.csv', '')
    return process_csv_with_chunked_pandas(file_path, file_key, all_categorical_values)


def extract_ip_features(ip_series):
    """Extracts network-related features from a series of IP addresses."""

    def parse_ip(ip_str):
        if pd.isna(ip_str) or ip_str in ['-', ''] or ':' in str(ip_str): return [0, 0, 0, 0, 0]
        try:
            octets = str(ip_str).split('.')
            if len(octets) != 4: return [0, 0, 0, 0, 0]
            first = int(octets[0])
            is_private = 1 if first == 10 or first == 192 or (first == 172 and 16 <= int(octets[1]) <= 31) else 0
            is_broadcast = 1 if ip_str == '255.255.255.255' else 0
            is_multicast = 1 if 224 <= first <= 239 else 0
            is_localhost = 1 if first == 127 else 0
            return [is_private, is_broadcast, is_multicast, first, is_localhost]
        except (ValueError, IndexError):
            return [0, 0, 0, 0, 0]

    ip_features = ip_series.apply(parse_ip)
    return pd.DataFrame(ip_features.tolist(), columns=Config.IP_FEATURES_BASE, index=ip_series.index)


def _clean_and_expand_labels(df):
    """Cleans and expands the 'label' and 'detailed-label' columns into structured target columns."""
    has_label_col = 'label' in df.columns and 'detailed-label' in df.columns
    if has_label_col:
        combined_str = (df['label'].astype(str).fillna('') + ' ' + df['detailed-label'].astype(str).fillna(
            '')).str.lower().str.strip()
    else:
        combined_str = pd.Series([''] * len(df))

    # Remove literal "(empty)" markers that appear in some files
    combined_str = combined_str.str.replace(r'\(empty\)', '', regex=True).str.strip()

    df['label'] = 'Benign'
    df.loc[combined_str.str.contains('malicious', na=False), 'label'] = 'Malicious'
    df['attack_type'] = 'Benign'
    df['attack_subtype'] = pd.NA
    df['malware_family'] = pd.NA

    malicious_mask = df['label'] == 'Malicious'
    if not malicious_mask.any():
        if 'detailed-label' in df.columns: df = df.drop(columns=['detailed-label'], errors='ignore')
        return df

    malicious_labels = combined_str[malicious_mask]
    patterns = {
        'attack_type': {'C&C': re.compile(r'c&c'), 'DDoS': re.compile(r'ddos'),
                        'PortScan': re.compile(r'partofahorizontalportscan|portscan'), 'Attack': re.compile(r'attack'),
                        'FileDownload': re.compile(r'filedownload')},
        'malware_family': {'Mirai': re.compile(r'mirai'), 'Okiru': re.compile(r'okiru'), 'Torii': re.compile(r'torii')},
        'attack_subtype': {'HeartBeat': re.compile(r'heartbeat'), 'FileDownload': re.compile(r'filedownload')}
    }

    def parse_label(label_str):
        label_str = re.sub(r'\(empty\)|-|\bmalicious\b|\bbenign\b', '', label_str).strip()
        parsed = {'attack_type': set(), 'attack_subtype': set(), 'malware_family': set()}
        for category, cat_patterns in patterns.items():
            for key, pattern in cat_patterns.items():
                if pattern.search(label_str): parsed[category].add(key)

        # ✅ FIX: Check if this is a malware family FIRST - they should NOT be attack types
        primary_type = 'Unknown'
        if parsed['malware_family']:
            # If malware family detected, the attack_type should be "Malware", not the family name
            primary_type = 'Malware'  # ← CHANGED: Use generic "Malware" instead of family name
        elif 'PortScan' in parsed['attack_type']:
            primary_type = 'PortScan'
        elif 'C&C' in parsed['attack_type']:
            primary_type = 'C&C'
        elif 'DDoS' in parsed['attack_type']:
            primary_type = 'DDoS'
        elif 'FileDownload' in parsed['attack_type']:
            primary_type = 'FileDownload'
        elif 'Attack' in parsed['attack_type']:
            primary_type = 'Attack'

        subtype = ','.join(sorted(list(parsed['attack_subtype']))) if parsed['attack_subtype'] else pd.NA
        family = list(parsed['malware_family'])[0] if parsed['malware_family'] else pd.NA
        if primary_type == 'Unknown' and label_str: primary_type = label_str.replace(' ', '').title()
        return primary_type, subtype, family

    parsed_results = malicious_labels.apply(parse_label)
    df.loc[malicious_mask, 'attack_type'] = [res[0] for res in parsed_results]
    df.loc[malicious_mask, 'attack_subtype'] = [res[1] for res in parsed_results]
    df.loc[malicious_mask, 'malware_family'] = [res[2] for res in parsed_results]
    if 'detailed-label' in df.columns: df = df.drop(columns=['detailed-label'], errors='ignore')
    return df


def _preprocess_partition(df, all_categorical_values):
    """Fully robust preprocessing function that handles inconsistent columns and dirty data."""

    rename_dict = {'id.orig_h': 'id_orig_h', 'id.orig_p': 'id_orig_p', 'id.resp_h': 'id_resp_h',
                   'id.resp_p': 'id_resp_p'}
    df = df.rename(columns=rename_dict)

    # ===== CRITICAL: FORCE CORRECT DATA TYPES FOR ALL COLUMNS =====

    # 1. String columns (IPs, UIDs, etc.)
    string_cols = ['uid', 'id_orig_h', 'id_resp_h', 'proto', 'service', 'conn_state',
                   'history', 'Source_Folder', 'label', 'detailed-label']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', '<NA>'], '')

    # 2. Boolean/categorical columns that should be strings
    bool_cols = ['local_orig', 'local_resp']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', '<NA>'], '')

    # 3. Port columns (force to int)
    if 'id_orig_p' in df.columns:
        df['id_orig_p'] = pd.to_numeric(df['id_orig_p'], errors='coerce').fillna(0).astype(int)
    if 'id_resp_p' in df.columns:
        df['id_resp_p'] = pd.to_numeric(df['id_resp_p'], errors='coerce').fillna(0).astype(int)

    # 4. IP addresses (force to clean strings)
    if 'id_orig_h' in df.columns:
        df['id_orig_h'] = df['id_orig_h'].fillna('0.0.0.0').astype(str).str.strip()
    if 'id_resp_h' in df.columns:
        df['id_resp_h'] = df['id_resp_h'].fillna('0.0.0.0').astype(str).str.strip()

    # 5. Timestamp (force to numeric)
    if 'ts' in df.columns:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce').fillna(0)

    # Now proceed with label cleaning
    if 'label' in df.columns:
        df = _clean_and_expand_labels(df)

    # Process numeric columns
    numeric_cols_to_process = set(Config.BASE_NUMERICAL_FEATURES + Config.SKEWED_NUMERICAL_FEATURES)
    for col in numeric_cols_to_process:
        if col in df.columns:
            df[f'{col}_was_missing'] = pd.to_numeric(df[col], errors='coerce').isnull().astype(np.int8)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
            if col in Config.SKEWED_NUMERICAL_FEATURES:
                df[col] = np.log1p(df[col])
        else:
            df[col] = 0
            df[f'{col}_was_missing'] = 1

    for col in Config.CATEGORICAL_LABEL_ENCODE:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).replace(['', '-', 'nan', 'None'], 'unknown')
            df[col] = pd.Categorical(df[col]).codes.astype(np.int32)
        else:
            df[col] = 0

    for col in Config.CATEGORICAL_ONE_HOT_ENCODE:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).replace(['', '-', 'nan', 'None'], 'unknown')
            for category in all_categorical_values.get(col, []):
                df[f'{col}_{category}'] = (df[col] == category).astype(np.int8)

    df['is_port_23'] = (df.get('id_resp_p', 0) == 23).astype(np.int8)
    df['is_port_22'] = (df.get('id_resp_p', 0) == 22).astype(np.int8)
    df['is_S0_state'] = df.get('conn_state_S0', 0)

    total_bytes = df.get('orig_bytes', 0) + df.get('resp_bytes', 0)
    total_packets = df.get('orig_pkts', 0) + df.get('resp_pkts', 0)
    df['upload_ratio'] = df.get('orig_bytes', 0) / (total_bytes + 1e-9)
    df['bytes_per_packet'] = total_bytes / (total_packets + 1e-9)

    safe_duration = np.expm1(df.get('duration', 0)).clip(lower=0.001)
    df['packet_rate'] = (total_packets / safe_duration).clip(upper=10000)
    df['is_scanning_signature'] = ((df['is_port_23'] == 1) & (df['is_S0_state'] == 1)).astype(np.int8)
    df['suspicious_score'] = df['is_port_23'] * 40 + df['is_S0_state'] * 30 + df['is_port_22'] * 25

    if 'id_resp_h' in df.columns:
        resp_ip_features = extract_ip_features(df['id_resp_h'])
        resp_ip_features.columns = ['resp_' + c for c in resp_ip_features.columns]
        df = pd.concat([df, resp_ip_features], axis=1)

    if 'id_orig_h' in df.columns:
        orig_ip_features = extract_ip_features(df['id_orig_h'])
        orig_ip_features.columns = ['orig_' + c for c in orig_ip_features.columns]
        df = pd.concat([df, orig_ip_features], axis=1)

    # ✅ FIX: Drop original categorical columns after one-hot encoding
    columns_to_drop = ['uid', 'conn_state', 'proto', 'local_orig', 'local_resp']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    return df

def process_csv_with_chunked_pandas(file_path, file_key, all_categorical_values):
    """Reads a CSV in chunks, processes each chunk, and saves the result as a Parquet file."""
    output_path = os.path.join(Config.ENGINEERED_SPLIT_DIR, f"{file_key}.parquet")
    chunks_list = []

    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines='skip', comment='#'):
            chunk['Source_Folder'] = file_key
            processed = _preprocess_partition(chunk, all_categorical_values)

            ground_truth_family = Config.FILENAME_TO_FAMILY_MAP.get(file_key, 'Unknown')
            if 'malware_family' not in processed.columns:
                processed['malware_family'] = ground_truth_family  # Use ground truth directly
            processed['malware_family'] = processed['malware_family'].fillna(ground_truth_family)

            # Should be (delete lines 222 and 224):
            # Only apply ground truth if the file is ACTUALLY benign AND labels are truly missing
            if ground_truth_family == 'Benign':
                # Only set label if it's NULL or Unknown - don't override parsed values
                if 'label' in processed.columns:
                    processed.loc[processed['label'].isin(['Unknown', None, '']), 'label'] = 'Benign'
                else:
                    processed['label'] = 'Benign'

                # Same for attack_type
                if 'attack_type' in processed.columns:
                    processed.loc[processed['attack_type'].isin(['Unknown', None, '']), 'attack_type'] = 'Benign'
                else:
                    processed['attack_type'] = 'Benign'

            chunks_list.append(processed)

        if chunks_list:
            final_df = pd.concat(chunks_list, ignore_index=True)
            final_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
            tqdm.write(f"    ✅ Processed {os.path.basename(file_path)}: {len(final_df):,} rows")
            return len(final_df)
    except Exception as e:
        tqdm.write(f"    ❌ FAILED to process {os.path.basename(file_path)}: {e}")
        return 0
    return 0


def build_engineered_dataset():
    """Builds the final dataset from all raw CSVs."""

    if os.path.exists(Config.ENGINEERED_SPLIT_DIR):
        print(f"🗑️  Removing old intermediate directory: {Config.ENGINEERED_SPLIT_DIR}")
        shutil.rmtree(Config.ENGINEERED_SPLIT_DIR)
    os.makedirs(Config.ENGINEERED_SPLIT_DIR)

    print("\n" + "=" * 70)
    print("🚀 STARTING DATASET BUILD PIPELINE (Pandas)")
    print("=" * 70)

    csv_files = glob.glob(os.path.join(Config.RAW_DIR_ORIGINAL, '*.csv'))
    if not csv_files:
        print(f"❌ Error: No CSV files found in '{Config.RAW_DIR_ORIGINAL}'")
        return

    all_categorical_values = scan_categorical_values(csv_files)

    print("\n" + "=" * 70)
    print("🔍 PROCESSING RAW FILES (PARALLEL)")
    print("=" * 70)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        args_list = [(fp, all_categorical_values) for fp in csv_files]
        list(tqdm(executor.map(process_single_csv, args_list), total=len(args_list), desc="Processing files"))

    print("\n" + "=" * 70)
    print("🔍 COMBINING PROCESSED FILES")
    print("=" * 70)

    split_files = glob.glob(os.path.join(Config.ENGINEERED_SPLIT_DIR, '*.parquet'))
    if not split_files:
        print("❌ No processed files found to combine.")
        return

    all_samples = [pd.read_parquet(f) for f in tqdm(split_files, desc="Loading processed files")]
    final_df = pd.concat(all_samples, ignore_index=True)
    print(f"✓ Final dataset created with {len(final_df):,} rows.")

    print("\n⏳ Building and saving final feature list...")
    target_cols = [Config.TARGET_COL, Config.DETAILED_TARGET_COL, Config.FAMILY_TARGET_COL, 'attack_subtype']
    metadata_cols = ['id_orig_h', 'id_orig_p', 'id_resp_h', 'id_resp_p', 'Source_Folder']

    for col in metadata_cols + target_cols:
        if col not in final_df.columns:
            final_df[col] = None

    final_feature_list = [col for col in final_df.columns if col not in metadata_cols + target_cols]
    final_column_order = metadata_cols + final_feature_list + target_cols
    final_df = final_df.reindex(columns=final_column_order)

    joblib.dump(final_feature_list, Config.FEATURE_LIST_PATH)
    print(f"✓ Feature list saved with {len(final_feature_list)} features.")

    print("\n⏳ Saving final Parquet file...")
    final_df.to_parquet(Config.ENGINEERED_DATA_PATH, compression='snappy', index=False)
    print(f"✓ Parquet saved to {Config.ENGINEERED_DATA_PATH}")

    print("\n🗑️  Cleaning up intermediate files...")
    shutil.rmtree(Config.ENGINEERED_SPLIT_DIR)
    print(f"✅ Removed intermediate directory.")
    print("=" * 70)


if __name__ == "__main__":
    build_engineered_dataset()
