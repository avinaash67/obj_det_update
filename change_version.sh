#!/bin/bash
sed "s/latest/$1/g" kub_dep.yaml > new_version.yml
