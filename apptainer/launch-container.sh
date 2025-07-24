export repo='/mmfs1/home/alexeyy/storage/CTF-for-Science/models/llmtime'
export cache="/mmfs1/home/alexeyy/storage/.cache"
apptainer run --nv --cwd "/app/code" --bind "$repo":"/app/code" "$repo"/apptainer/gpu.sif
#apptainer run --nv --cwd "/app/code" --overlay "$repo"/apptainer/overlay.img --no-home --contain --bind "$repo":"/app/code" "$repo"/apptainer/gpu.sif

# fuser -v overlay.img
# kill -9 <pid>
# fsck.ext3 overlay.img
