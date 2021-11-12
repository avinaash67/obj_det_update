#!/bin/bash
sed "s/latest/$1/g" kubernetes_dep.yaml > new_version.yml
