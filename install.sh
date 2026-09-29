# Keep Apptainer's temporary build and cache files off the default /tmp, which can be too small
# for a large SIF build.
APPTAINER_BUILD_ROOT="${APPTAINER_BUILD_ROOT:-${HOME}/scratch/apptainer-s2p-project}"
mkdir -p "${APPTAINER_BUILD_ROOT}/tmp" "${APPTAINER_BUILD_ROOT}/cache"
export APPTAINER_TMPDIR="${APPTAINER_BUILD_ROOT}/tmp"
export APPTAINER_CACHEDIR="${APPTAINER_BUILD_ROOT}/cache"
export TMPDIR="${APPTAINER_TMPDIR}"

# Remove the container files if they exist
rm containers/s2p.sif containers/s2p_custom_dependencies.overlay

# # Build the image
apptainer build containers/s2p.sif containers/s2p.def

# # Create a small overlay to install custom dependencies
apptainer overlay create --size 128 containers/s2p_custom_dependencies.overlay

# Install the custom dependencies.
#
# The overlay is mounted read-write here, which takes an exclusive lock: this will refuse to run
# ("currently in use by another process") while any job has the same file mounted, even read-only.
# That refusal is protective -- writing into the ext3 image under a running job's feet corrupts it.
# So take the overlay path as an argument: point it at a copy while jobs are live, and swap the copy
# into place once they finish.
OVERLAY="${1:-containers/s2p_custom_dependencies.overlay}"
echo "Installing into overlay: ${OVERLAY}"

apptainer exec \
	--overlay "${OVERLAY}" \
	--bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
	containers/s2p.sif \
	bash -c "
		/opt/venv/bin/python3 -m pip install \
			-e ${PROJECT_ROOT}/agents/squeeze2plan --no-build-isolation \
			-e ${PROJECT_ROOT}/agents/tdmpc2 --no-build-isolation \
			-e ${PROJECT_ROOT}/agents/repo --no-build-isolation \
			-e ${PROJECT_ROOT}/agents/minco --no-build-isolation \
			-e ${PROJECT_ROOT}/agents/dino_wm --no-build-isolation \
			-e ${PROJECT_ROOT}/environments/custom_maniskill_tasks --no-build-isolation \
	"
