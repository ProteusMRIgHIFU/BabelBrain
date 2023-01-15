SCRIPT_DIR="${0:a:h}"
target="/mac/lib/"
echo ${SCRIPT_DIR}
OLD_DYLD=$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH="${SCRIPT_DIR}${target}":$DYLD_LIBRARY_PATH
"$SCRIPT_DIR/mac/bin/elastix"  -f "$1" -m "$2" -out "$3"  -p "$SCRIPT_DIR/rigid.txt"
output=$?
export DYLD_LIBRARY_PATH=$OLD_DYLD
echo "elastix ended with code " $output
exit output