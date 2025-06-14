#!/bin/bash

# This script unpacks the data archives.

echo "Extracting data..."

tar -xzf CoFT/CoFT/data/epilepsy.tar.gz -C CoFT/CoFT/data/
tar -xzf CoFT/CoFT/data/har.tar.gz -C CoFT/CoFT/data/
tar -xzf CoFT/CoFT/data/sleep.tar.gz -C CoFT/CoFT/data/
tar -xzf CoFT/CoFT/data/sleepedf.tar.gz -C CoFT/CoFT/data/

echo "Data extraction complete." 