import pandas as pd
import numpy as np
import sys

# --- CONFIGURATION ---
# Change this to match your actual python file name (without .py)
FRAMEWORK_FILENAME = "framework_29" 
DATASET_FILE = "dataset_29.csv"
TARGET_FILE = "target_29.csv"

# 1. Import your framework dynamically
try:
    framework_module = __import__(FRAMEWORK_FILENAME)
    print(f"✅ Successfully imported {FRAMEWORK_FILENAME}.py")
except ImportError:
    print(f"❌ ERROR: Could not find {FRAMEWORK_FILENAME}.py in this folder.")
    sys.exit()

# 2. Mock the Arguments
# Your framework expects an object with an 'eval_file_path' attribute
class MockArgs:
    def __init__(self, path):
        self.eval_file_path = path

args = MockArgs(DATASET_FILE)

# 3. Run Your Framework
print(f"Running framework on {DATASET_FILE}...")
try:
    # We call the main() function from your script
    calculated_targets = framework_module.main(args)
    print(f"✅ Framework executed successfully. Generated {len(calculated_targets)} predictions.")
except Exception as e:
    print(f"❌ CRASH: Your framework script crashed.")
    print(e)
    sys.exit()

# 4. Compare with the Answer Key
print(f"Loading true answers from {TARGET_FILE}...")
df_true = pd.read_csv(TARGET_FILE)

# Check if 'target02' exists, otherwise assume it's the second column
if 'target02' in df_true.columns:
    true_values = df_true['target02'].values
else:
    print("Warning: Column 'target02' not found by name. Using the 2nd column.")
    true_values = df_true.iloc[:, 1].values

# Convert lists to numpy for math
pred_values = np.array(calculated_targets)
diff = np.abs(true_values - pred_values)

# 5. The Verdict
max_error = np.max(diff)
mean_error = np.mean(diff)

print("-" * 30)
print(f"MAX ERROR:  {max_error:.10f}")
print(f"MEAN ERROR: {mean_error:.10f}")
print("-" * 30)

if max_error < 1e-10:
    print("🎉 SUCCESS! Your framework is PERFECT.")
    print("It reproduces the training targets exactly.")
    print("You are ready to submit.")
else:
    print("⚠️ FAIL. The numbers do not match.")
    print("Possible reasons:")
    print("1. The feature indices (80, 180, 200, 146) are shifted (try adding +1 or -1).")
    print("2. The rule logic needs a slight tweak.")