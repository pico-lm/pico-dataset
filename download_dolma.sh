DATA_DIR="data/dolma"
PARALLEL_DOWNLOADS=10
DOLMA_VERSION="v1_7"

# Clone the repository if you haven't already
# git clone https://huggingface.co/datasets/allenai/dolma
# if not exists, then clone
if [ ! -d "dolma" ]; then
  git clone https://huggingface.co/datasets/allenai/dolma
fi

# Create the directory if it doesn't exist
mkdir -p "${DATA_DIR}"

# Function to check if the file already exists and resume download if incomplete
download_if_incomplete() {
  local url=$1
  local file_path="${DATA_DIR}/$(basename ${url})"

  if [ -f "${file_path}" ]; then
    wget --spider -q "${url}"
    if [ $? -eq 0 ]; then
      remote_size=$(wget --spider "${url}" 2>&1 | grep Length | awk '{print $2}')
      local_size=$(stat --format="%s" "${file_path}")

      if [ "${local_size}" -eq "${remote_size}" ]; then
        echo "File ${file_path} is fully downloaded, skipping."
        return 0
      else
        echo "File ${file_path} is incomplete, resuming download..."
      fi
    fi
  fi
#   wget -c -q -P "${DATA_DIR}" "${url}"
  echo "$2/$3 files downloaded."
}

export -f download_if_incomplete
export DATA_DIR

# Count total number of URLs
total_urls=$(wc -l < "dolma/urls/${DOLMA_VERSION}.txt")
counter=0

# Function to update and display progress
progress_bar() {
  ((counter++))
  echo -ne "Progress: $counter/$total_urls\r"
}

export -f progress_bar

# Download files and update the progress bar
cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" bash -c 'download_if_incomplete "$0" "$counter" "$total_urls"; progress_bar'
