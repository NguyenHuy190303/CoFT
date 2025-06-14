#!/bin/bash

# This script unpacks the data archives.

echo "Extracting data..."

tar -xzf data/epilepsy.tar.gz -C data/
tar -xzf data/har.tar.gz -C data/
tar -xzf data/sleep.tar.gz -C data/
tar -xzf data/sleepedf.tar.gz -C data/

echo "Data extraction complete." 