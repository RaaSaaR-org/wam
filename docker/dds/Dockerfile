# DDS conformance image for the Unitree G1 transport (T-21, FR-06, OD-08).
#
# Contract:
# - linux/arm64 Ubuntu 22.04 — the same OS/arch family as the G1 EDU4 onboard Jetson Orin,
#   so nothing here is emulated and the wire format is the real one.
# - Builds Eclipse CycloneDDS (C core, pinned tag) and the CycloneDDS Python binding from
#   source, then unitree_sdk2py (pinned commit) as an EDITABLE install: the vendor
#   setup.py ships no package_data, so a regular install would DROP
#   unitree_sdk2py/utils/lib/crc_aarch64.so and every CRC would silently fall back /
#   crash. Editable keeps the checkout (and that .so) on disk.
# - The WAM repo is NOT copied in. It is bind-mounted at /wam at run time; PYTHONPATH
#   points at /wam/src so `wam.robot.g1_transport` is the working-tree copy.
# - No robot, no network peers: discovery stays inside the container on `lo`.
#
# Build:  docker build --platform linux/arm64 -t wam-dds-conformance docker/dds
# Run:    ./run.sh   (see README.md)

# ---------------------------------------------------------------------------------------
# Stage 1 — builder: compilers, CMake, source trees. None of this reaches the final image.
# ---------------------------------------------------------------------------------------
FROM --platform=linux/arm64 ubuntu:22.04 AS builder

# CycloneDDS 0.10.2 (tag 0.10.2) — the version unitree_sdk2py pins its Python binding to.
ARG CYCLONEDDS_REF=9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd
ARG CYCLONEDDS_PY_VERSION=0.10.2
# unitree_sdk2py master @ 2026-07 (v1.0.1); no tags upstream, so pin the commit.
ARG UNITREE_SDK2PY_REF=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# --- CycloneDDS C core -----------------------------------------------------------------
# BUILD_IDLC=ON is required: the Python binding compiles its idlpy plugin against it.
RUN git clone --quiet https://github.com/eclipse-cyclonedds/cyclonedds.git /src/cyclonedds \
    && git -C /src/cyclonedds checkout --quiet "${CYCLONEDDS_REF}" \
    && cmake -S /src/cyclonedds -B /src/cyclonedds/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds \
        -DBUILD_IDLC=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DENABLE_SECURITY=OFF \
        -DENABLE_SSL=OFF \
    && cmake --build /src/cyclonedds/build --target install --parallel "$(nproc)" \
    && rm -rf /src/cyclonedds

ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV PATH=/opt/venv/bin:$PATH

# --- Python environment ----------------------------------------------------------------
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade "pip==24.2" "setuptools==75.1.0" "wheel==0.44.0"

# The Python binding has no arm64 wheel; it builds against CYCLONEDDS_HOME.
RUN pip install --no-cache-dir --no-binary :all: "cyclonedds==${CYCLONEDDS_PY_VERSION}"

# WAM runtime deps. `wam.robot` itself is torch-free (numpy + pydantic); pyyaml comes in
# via wam/__init__.py -> wam.interfaces.versioning. Nothing else from the repo is needed.
RUN pip install --no-cache-dir "numpy==1.26.4" "pydantic==2.9.2" "pyyaml==6.0.2"

# --- unitree_sdk2py --------------------------------------------------------------------
# --no-deps: its install_requires pulls opencv-python, which nothing in the package
# actually imports (verified: zero `import cv2`) and which costs ~200 MB on arm64.
# -e: keeps utils/lib/crc_aarch64.so reachable (see header note).
RUN git clone --quiet https://github.com/unitreerobotics/unitree_sdk2py.git /opt/unitree_sdk2py \
    && git -C /opt/unitree_sdk2py checkout --quiet "${UNITREE_SDK2PY_REF}" \
    && rm -rf /opt/unitree_sdk2py/.git /opt/unitree_sdk2py/example \
    && pip install --no-cache-dir --no-build-isolation --no-deps -e /opt/unitree_sdk2py

# ---------------------------------------------------------------------------------------
# Stage 2 — runtime: interpreter + the two artefacts. No compilers, no source trees.
# ---------------------------------------------------------------------------------------
FROM --platform=linux/arm64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# iproute2 only so the conformance report can name the loopback interface it bound to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/cyclonedds /opt/cyclonedds
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/unitree_sdk2py /opt/unitree_sdk2py

ENV CYCLONEDDS_HOME=/opt/cyclonedds \
    LD_LIBRARY_PATH=/opt/cyclonedds/lib \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/wam/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /wam

# conformance.py is read from the bind mount, not baked in, so it can be edited without a
# rebuild. Run with:  docker run --rm -v <repo>:/wam:ro wam-dds-conformance
CMD ["python3", "/wam/docker/dds/conformance.py"]
