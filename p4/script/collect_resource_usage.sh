#!/bin/sh

mkdir -p ../resource
for n in "elastic_sketch" "mrac" "sample" "neufsd" "davinci"; do
    cp $SDE/build/p4-build/$n/$n/tofino/pipe/logs/mau.resources.log ../resource/${n}_resource.log
done
