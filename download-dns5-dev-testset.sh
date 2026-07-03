#!/usr/bin/bash

URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_dev_testset.zip"

echo "Downloading dev testset (~4GB)..."
curl -L -o V5_dev_testset.zip https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_dev_testset.zip

echo "Unpacking..."
unzip V5_dev_testset.zip