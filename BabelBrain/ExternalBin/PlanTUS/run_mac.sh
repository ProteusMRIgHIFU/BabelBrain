#!/bin/zsh
# This script is used to run the PlanTUS_wrapper.py script in a Python virtual environment
# Usage: ./run_mac.sh <path_to_simnibs_env> <path_to_TUSPlan> <input_T1w> <input_mesh> <input_mask> <TxConfig>
# Check if the correct number of arguments is provided
if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <path_to_simnibs_env> <path_to_TUSPlan> <input_T1w> <input_mesh> <input_mask> <TxConfig>"
    exit 1
fi
# Activate the Python virtual environment and run the PlanTUS_wrapper.py script
# Check if the simnibs_env directory exists
echo "Activating virtual environment from $1/simnibs_env/bin/activate"
source "$1/simnibs_env/bin/activate"
echo "running $2/PlanTUS_wrapper.py with arguments: $3 $4 $5 $6 --skip_wb_view $7 $8"
python "$2/PlanTUS_wrapper.py" "$3" "$4" "$5" "$6" --skip_wb_view $7 $8