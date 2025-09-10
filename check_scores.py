#!/usr/bin/env python3

import sys
import pandas as pd


def main(file1, file2):
    try:
        # Read Excel files from score_all/score_en/ directory
        df1 = pd.read_excel("score_all/score_en/" + file1)
        df2 = pd.read_excel("score_all/score_en/" + file2)

        # Check if dataframes are identical
        if df1.equals(df2):
            print("The two Excel files are identical.")
        else:
            print("The two Excel files differ.")
            print("\nDifferences found:\n")
            print(df1 == df2)

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Make sure files are in the score_all/score_en/ directory")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./check_scores.py file1.xlsx file2.xlsx")
        print("Files should be in score_all/score_en/ directory")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    main(file1, file2)
