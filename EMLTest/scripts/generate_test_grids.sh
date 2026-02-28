#!/bin/bash

# Read base input file
BASE_FILE="${1:-ene_psb3_blyp_631g.in}"
# Check if BASE_FILE exists
if [ ! -f "$BASE_FILE" ]; then
    echo "✗ Error: Base file not found: $BASE_FILE"
    exit 1
fi

# Extract directory path from BASE_FILE
BASE_DIR=$(dirname "$BASE_FILE")
OUTPUT_DIR="$BASE_DIR/test_grids"
mkdir -p "$OUTPUT_DIR"
echo "✓ Base file: $BASE_FILE"
echo "✓ Output directory: $OUTPUT_DIR"

# Format: (radial,angular)
configs=(
    "5,194"
    "10,194"
    "15,194"
    "20,194"
    "25,194"
    "30,194"
    "35,194"
    "40,194"
    "45,194"
    "50,194"
    "55,194"
    "60,194"
    "65,194"
    "50,14"
    "50,26"
    "50,38"
    "50,50"
    "50,74"
    "50,86"
    "50,110"
    "50,146"
    "50,170"
    "50,230"
    "50,266"
)

# Extract everything before DFT line (handles lines with leading spaces)
HEAD=$(sed '/DFT/,$d' "$BASE_FILE")

# Extract everything after DFT line (including the DFT line itself, then remove it)
TAIL=$(sed '1,/DFT/d' "$BASE_FILE")

# Generate test files
for config in "${configs[@]}"; do
    IFS=',' read -r rad ang <<< "$config"
    output_file="$OUTPUT_DIR/ene_psb3_blyp_631g_${rad}_${ang}.in"
    echo "Generating $output_file"
    echo "$HEAD" > "$output_file"
    echo "DFT SG1 IRADTEMP=$rad LEBEDEV_TYPE=$ang BLYP 6-31G" >> "$output_file"
    echo "$TAIL" >> "$output_file"
done

echo "Generated ${#configs[@]} test configuration files in $OUTPUT_DIR/"
