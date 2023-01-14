SCRIPT_DIR="$(dirname "$(realpath "$0")")"
target="/linux/lib/"
echo ${SCRIPT_DIR}
OLD_LD=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${SCRIPT_DIR}${target}":$LD_LIBRARY_PATH
"$SCRIPT_DIR/linux/bin/elastix"  -f "$1" -m "$2" -out "$3"  -p "$SCRIPT_DIR/rigid.txt"
output=$?
export LD_LIBRARY_PATH=$OLD_LD
echo "elastix ended with code " $output
exit $output