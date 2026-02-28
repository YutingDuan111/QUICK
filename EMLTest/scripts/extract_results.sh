#!/bin/bash

TEST_DIR="${1:-test_grids}"
OUTPUT_CSV="$TEST_DIR/grid_results.csv"

if [ ! -d "$TEST_DIR" ]; then
    echo "✗ Error: Test directory not found: $TEST_DIR"
    exit 1
fi

echo "✓ Test directory: $TEST_DIR"
echo "✓ Output CSV: $OUTPUT_CSV"
echo ""

echo "RadialPoints,AngularPoints,TotalGridPoints,TotalEnergy,AlphaElectronDensity" > "$OUTPUT_CSV"

# Process each .out file
for output_file in "$TEST_DIR"/ene_psb3_*.out; do
    if [ ! -f "$output_file" ]; then
        echo "✗ No .out files found in $TEST_DIR"
        exit 1
    fi
    
    filename=$(basename "$output_file" .out)
    
    # Parse radial and angular points from filename
    radial=$(echo "$filename" | rev | cut -d_ -f2 | rev)
    angular=$(echo "$filename" | rev | cut -d_ -f1 | rev)
    
    # Extract FINAL GRID POINTS (handle the | character)
    total_grid=$(grep "FINAL GRID POINTS" "$output_file" | awk '{print $NF}')
    
    # Extract TOTAL ENERGY (get the value after =)
    energy=$(grep "TOTAL ENERGY" "$output_file" | awk -F= '{print $NF}' | xargs)
    
    # Extract ALPHA ELECTRON DENSITY
    alpha_density=$(grep "ALPHA ELECTRON DENSITY" "$output_file" | awk -F= '{print $NF}' | xargs)
    
    # Handle empty fields
    total_grid=${total_grid:-"N/A"}
    energy=${energy:-"N/A"}
    alpha_density=${alpha_density:-"N/A"}
    
    # Write the data row to CSV
    echo "$radial,$angular,$total_grid,$energy,$alpha_density" >> "$OUTPUT_CSV"
done

echo "=========================================="
echo "✓ Results saved to: $OUTPUT_CSV"
echo "=========================================="
