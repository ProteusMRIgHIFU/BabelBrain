#!/bin/zsh
# This script is used to run the MeshConv.py script in a Python virtual environment
# Usage: ./run_mac.sh <path_to_simnibs_env> <path_to_script> <input_file> <output_file> <mesh_type>
# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <path_to_simnibs_env> <path_to_script> <input_file> <output_file> <mesh_type>"
    exit 1
fi
# Activate the Python virtual environment and run the MeshConv.py script
# Check if the simnibs_env directory exists
source "$1/simnibs_env/bin/activate"
python "$2/MeshConv.py" "$3" "$4" "$5" 