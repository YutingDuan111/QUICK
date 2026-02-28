#!/bin/bash

# Allow specifying the test directory as argument, or use default
TEST_DIR="${1:-test_grids}"

# Check if TEST_DIR exists
if [ ! -d "$TEST_DIR" ]; then
    echo "✗ Error: Test directory not found: $TEST_DIR"
    echo "Usage: $0 [path/to/test_grids]"
    exit 1
fi


# Run QUICK from test_grids directory
cd "$TEST_DIR" || exit 1
count=0
for input_file in ene_psb3_*.in; do
    echo "Running: $input_file"
    quick "$input_file"
    ((count++))
    echo ""
done

echo "=========================================="
echo "✓ Completed: $count test cases"
echo "✓ Output files (.out) are in: $TEST_DIR"
echo "=========================================="
