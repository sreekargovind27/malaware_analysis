import pandas as pd

df = pd.read_parquet('data/engineered_features/final_features.parquet')

print("local_orig dtype:", df['local_orig'].dtype)
print("local_resp dtype:", df['local_resp'].dtype)
print("\nlocal_orig unique:", df['local_orig'].unique()[:10])
print("local_resp unique:", df['local_resp'].unique()[:10])
print("\nAll zeros?")
print("local_orig:", (df['local_orig'] == 0).all())
print("local_resp:", (df['local_resp'] == 0).all())