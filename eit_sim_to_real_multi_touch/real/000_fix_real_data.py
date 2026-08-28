import pandas as pd

in_path = "/home/kiyanoush/Projects/eit_sim_to_real_multi_touch/data/eit_windows/eit_robot_fused_1760625340.csv"
out_path = "/home/kiyanoush/Projects/eit_sim_to_real_multi_touch/data/eit_windows/eit_double_touch_data.csv"

df = pd.read_csv(in_path)
cols = list(df.columns)  # copy to list

def shift_left_header(cols_list, start_idx, placeholder):
    """
    Shift all column names left starting at start_idx.
    The last column name is replaced with 'placeholder'.
    This keeps the length identical to the original.
    """
    # Move each name one position left
    for i in range(start_idx, len(cols_list) - 1):
        cols_list[i] = cols_list[i + 1]
    # Put a placeholder in the last slot
    cols_list[-1] = placeholder

changes = []

# 1) Remove R1_tau7 (by shifting names left starting at its position)
if "R1_tau7" in cols:
    idx_r1 = cols.index("R1_tau7")
    shift_left_header(cols, idx_r1, "__extra_1")
    changes.append(f"Removed 'R1_tau7' at index {idx_r1}, shifted left, filled last with '__extra_1'.")
else:
    changes.append("R1_tau7 not found; no change.")

# 2) Remove R2_tau7 on the updated header
if "R2_tau7" in cols:
    idx_r2 = cols.index("R2_tau7")
    shift_left_header(cols, idx_r2, "__extra_2")
    changes.append(f"Removed 'R2_tau7' at index {idx_r2}, shifted left, filled last with '__extra_2'.")
else:
    changes.append("R2_tau7 not found; no change.")

# Assign corrected column names (length remains identical)
df.columns = cols

# Save
df.to_csv(out_path, index=False)
print(f"Saved corrected CSV to: {out_path}")
print("Changes:")
for c in changes:
    print(" -", c)

# Optional: quick sanity checks
print(f"Original number of columns: {len(list(pd.read_csv(in_path, nrows=0).columns))}")
print(f"New number of columns: {len(df.columns)}")