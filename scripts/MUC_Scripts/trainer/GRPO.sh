#!/bin/bash

# Set environment variables for CPU affinity
source /home/mzubillaga/.bashrc


CPUSET_FILE="/sys/fs/cgroup/cpuset/slurm/uid_${UID}/job_${SLURM_JOBID}/cpuset.cpus"

# Read the CPU set and expand any ranges (like 0-4)
CPUSET=$(cat "$CPUSET_FILE")
echo "Raw CPUSET from file: $CPUSET"

# Expand CPU ranges using a function
expand_cpu_list() {
    local cpulist=$1
    local expanded=""
    
    # Process comma-separated segments
    for segment in ${cpulist//,/ }; do
        # Check if it's a range (contains a dash)
        if [[ $segment == *-* ]]; then
            local start=${segment%-*}
            local end=${segment#*-}
            # Expand the range
            for ((i=start; i<=end; i++)); do
                expanded+="$i "
            done
        else
            # It's a single CPU
            expanded+="$segment "
        fi
    done
    
    echo $expanded
}

# Get expanded list of all CPUs
ALL_CPUS=($(expand_cpu_list "$CPUSET"))

# Check if we have enough CPUs
if [ ${#ALL_CPUS[@]} -lt 8 ]; then
    echo "Error: Need at least 8 CPUs, but only ${#ALL_CPUS[@]} assigned"
    exit 1
fi

# First 2 CPUs for background job
BG_CPUS="${ALL_CPUS[0]},${ALL_CPUS[1]},${ALL_CPUS[2]}"

# Next 4 CPUs for main job
MAIN_CPUS="${ALL_CPUS[3]},${ALL_CPUS[4]},${ALL_CPUS[5]},${ALL_CPUS[6]},${ALL_CPUS[7]}"

echo "Using CPUs: $BG_CPUS for background job, $MAIN_CPUS for main job"


# Start the background job using CPUs 0-1
(
    echo "Server"
    export CUDA_VISIBLE_DEVICES=""  # No GPU for background job
    conda activate evaluate_iterX
    taskset -c $BG_CPUS python scripts/MUC_Scripts/trainer/reward_server.py &
    #python scripts/MUC_Scripts/trainer/reward_server.py &
)

background_pid=$!
sleep 20  # Wait for the background job to start
echo "Martxan"
conda deactivate
# Start the main job using CPUs 2-5 and all GPUs
(
    echo "Main"
    export CUDA_VISIBLE_DEVICES="0,1,2,3"  # All GPUs
    source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
    python scripts/MUC_Scripts/trainer/probatu_server.py
    taskset -c $MAIN_CPUS accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/GRPO.py --model-path Model_JSONR1 --distributed --batch-size 4 --gradient-accumulation-steps 4
    #accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/GRPO.py --model-path Model_Proba --distributed

)

# Wait for all jobs to complete

kill $background_pid

# Wait for the background job to be properly terminated
wait $background_pid 2>/dev/null

echo "All jobs completed. Background job was terminated."