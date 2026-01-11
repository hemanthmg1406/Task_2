import pandas as pd
import numpy as np
import operator

# --- 1. SETUP: LOAD YOUR TRAINING DATA ---
# Replace with your actual filenames
id_num = "29" 
df_features = pd.read_csv(f"dataset_29.csv")
df_targets = pd.read_csv(f"target_29.csv")

# Extract the arrays
X_train = df_features.values
y_true = df_targets['target02'].values # We are testing Target 02

print(f"Loaded {len(X_train)} rows of Training Data.")

# --- 2. DEFINE THE LOGIC (COPIED FROM YOUR FRAMEWORK.PY) ---
# We verify the EXACT same functions you will submit.

def calc_segment_1(row):
    return -1.05 * row[180] - 0.85 * row[200] - 0.25 * row[146]

def calc_segment_2(row):
    return -0.45 * row[180] + 0.65 * row[200] - 0.45 * row[146]

def calc_segment_3(row):
    return -2.05 * row[180] + 0.05 * row[200] + 0.75 * row[146]

def calc_segment_4(row):
    return -1.35 * row[180] + 0.75 * row[200] - 1.55 * row[146]

# --- 3. EXECUTE THE LOGIC ---
y_pred = []

print("Running Logic on Training Data...")
for i in range(len(X_train)):
    row = X_train[i]
    
    # The "If/Else" Chain
    if row[80] <= 0.20:
        val = calc_segment_1(row)
    elif row[80] <= 0.50:
        val = calc_segment_2(row)
    elif row[80] <= 0.70:
        val = calc_segment_3(row)
    else:
        val = calc_segment_4(row)
        
    y_pred.append(val)

y_pred = np.array(y_pred)

# --- 4. CALCULATE ERROR ---
diff = np.abs(y_true - y_pred)
mae = np.mean(diff)
max_error = np.max(diff)

print("\n--- VERIFICATION RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.10f}")
print(f"Max Error in any row:      {max_error:.10f}")

if mae < 1e-10:
    print("\n✅ PASSED: The logic perfectly reproduces the training data.")
    print("You can verify that this script will work on the EVAL file.")
else:
    print("\n❌ FAILED: The logic does not match the training data.")
    print("Check if feature indices (80, 180, 200, 146) match the column numbers in your CSV.")