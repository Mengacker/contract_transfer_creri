#!/bin/bash
if [[ $PYPI_TOKEN ]]; then
  use $PYPI_TOKEN
fi
echo $PYPI_TOKEN
printf "[global]\n" > /etc/pip.conf
printf "extra-index-url =\n" >> /etc/pip.conf
printf "\thttps://$PYPI_TOKEN@pkgs.dev.azure.com/.../pypi/simple/\n" >> /etc/pip.conf
