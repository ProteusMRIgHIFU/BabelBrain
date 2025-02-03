SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
target="/mac/lib/"
echo ${SCRIPT_DIR}
OLD_DYLD=$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH="${SCRIPT_DIR}${target}":$DYLD_LIBRARY_PATH
"$SCRIPT_DIR/mac/bin/transformix" -loglevel off  -in "$1" -tp "$2" -out "$3"  > /dev/null
output=$?
export DYLD_LIBRARY_PATH=$OLD_DYLD
echo "transformix ended with code " $output
if [ $output -eq 0 ]; then
    true
else
    false
fi