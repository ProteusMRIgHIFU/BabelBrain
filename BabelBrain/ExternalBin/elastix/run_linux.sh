#!/bin/bash 
SCRIPT_DIR="$(dirname "$(realpath "$0")")" 
echo ${SCRIPT_DIR} 
OLD_LD=$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH="${SCRIPT_DIR}/linux/lib/":$LD_LIBRARY_PATH 
"$SCRIPT_DIR/linux/bin/elastix"  -f "$1" -m "$2" -out "$3"  -p  "$4"
output=$? 
export LD_LIBRARY_PATH=$OLD_LD 
echo "elastix lieux ended with code " $output 
exit $output 
