#!/bin/bash
echo cd $ARMNN_DIR/build/tests && cp ./TfMobileNet-Armnn /export/ && cp ../../tests/TfMobileNet-Armnn/Validation.txt /export/
if [[ -z "$1" ]]; then
  docker run --privileged -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -t armnn_aarch64:latest /bin/bash
else
  docker run --privileged -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -t $1 /bin/bash
fi
