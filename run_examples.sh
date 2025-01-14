#!/bin/bash

# Number of iterations
ITERATIONS=10

# Compile all examples first
echo "Compiling examples..."
cargo build --examples
if [ $? -ne 0 ]; then
    echo "Failed to compile examples"
    exit 1
fi

# Function to run an example
run_example() {
    local example=$1
    local iteration=$2
    
    echo "Running $example (iteration $iteration/$ITERATIONS)"
    # Run the compiled example directly
    ./target/debug/examples/$example
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "✓ $example completed successfully"
    else
        echo "✗ $example failed with status $status"
        return 1
    fi
}

# Main loop
for i in $(seq 1 $ITERATIONS); do
    echo "=== Iteration $i ==="
    
    # Run shared_array example
    run_example "shared_array" $i
    if [ $? -ne 0 ]; then
        echo "Error running shared_array example, stopping..."
        exit 1
    fi
    
    echo "------------------------"
done

echo "All examples completed successfully!"
