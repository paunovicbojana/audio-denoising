#!/usr/bin/bash

AZURE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
OUTPUT_PATH="./datasets_fullband"

mkdir -p "$OUTPUT_PATH/clean_fullband"
mkdir -p "$OUTPUT_PATH/noise_fullband"

CLEAN_BLOBS=(
    Track1_Headset/russian_speech.tgz
    Track1_Headset/VocalSet_48kHz_mono.tgz
)

NOISE_BLOBS=(
    noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
    datasets_fullband.impulse_responses_000.tar.bz2
)

echo "Downloading clean speech (~13GB)"
for BLOB in "${CLEAN_BLOBS[@]}"; do
    URL="$AZURE_URL/$BLOB"
    echo "-> $BLOB"
    curl -L "$URL" -o "$OUTPUT_PATH/clean_fullband/$(basename "$BLOB")"
done

echo "Unpacking clean speech "
for f in "$OUTPUT_PATH/clean_fullband/"*.tgz; do
    tar -C "$OUTPUT_PATH/clean_fullband/" -xzf "$f" && rm "$f"
done

echo "Downloading noise + impulse responses (~10GB)"
for BLOB in "${NOISE_BLOBS[@]}"; do
    URL="$AZURE_URL/$BLOB"
    echo "-> $BLOB"
    curl -L "$URL" -o "$OUTPUT_PATH/$(basename "$BLOB")"
done

echo "Unpacking noise"
for f in "$OUTPUT_PATH/"*.tar.bz2; do
    tar -C "$OUTPUT_PATH/" -xjf "$f" && rm "$f"
done

echo "Done"
du -sh "$OUTPUT_PATH/"