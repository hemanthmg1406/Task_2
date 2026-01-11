import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition"""
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


# --- IMPLEMENTATION OF THE FOUND RULES ---

def calc_segment_1(row):
    # Image Row 1: -1.05(feat_180) - 0.85(feat_200) - 0.25(feat_146)
    return -1.05 * row[180] - 0.85 * row[200] - 0.25 * row[146]

def calc_segment_2(row):
    # Image Row 2: -0.45(feat_180) + 0.65(feat_200) - 0.45(feat_146)
    return -0.45 * row[180] + 0.65 * row[200] - 0.45 * row[146]

def calc_segment_3(row):
    # Image Row 3: -2.05(feat_180) + 0.05(feat_200) + 0.75(feat_146)
    return -2.05 * row[180] + 0.05 * row[200] + 0.75 * row[146]

def calc_segment_4(row):
    # Image Row 4: -1.35(feat_180) + 0.75(feat_200) - 1.55(feat_146)
    return -1.35 * row[180] + 0.75 * row[200] - 1.55 * row[146]

def main(args):
    # 1. Define the Conditions based on feat_80
    # Note: The framework stops at the first True, so we order them ascending.
    
    cond_1 = (80, "<=", 0.20)
    cond_2 = (80, "<=", 0.50) # Implies 0.20 < feat_80 <= 0.50
    cond_3 = (80, "<=", 0.70) # Implies 0.50 < feat_80 <= 0.70
    cond_4 = None             # Implies feat_80 > 0.70 (The Else Case)

    # 2. Create the Pair List
    pair_list = [
        (cond_1, calc_segment_1),
        (cond_2, calc_segment_2),
        (cond_3, calc_segment_3),
        (cond_4, calc_segment_4)
    ]
    
    data_array = pd.read_csv(args.eval_file_path).values
    
    return framework(pair_list, data_array)

    
# Helper to show usage (matches original file structure)
def main_example(args):
    test_arr = np.ones((10,10))
    def calc1(arr): return arr[0]**2
    def calc2(arr): return arr[2] + arr[3]
    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)
    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print (predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)
    
    # Optional: Print results to verify
    # print(target02[:5])