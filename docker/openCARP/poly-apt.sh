#!/bin/bash
set -euo pipefail;

llvm_install() {
  local distribution="$(lsb_release -cs)";
  local pkg="llvm-${llvm_version}-dev clang-${llvm_version} clang-tools-${llvm_version} clang-${llvm_version}-doc libclang-common-${llvm_version}-dev libclang-${llvm_version}-dev libclang1-${llvm_version} clang-format-${llvm_version} python3-clang-${llvm_version} clangd-${llvm_version} clang-tidy-${llvm_version}";
  wget -q "https://apt.llvm.org/llvm-snapshot.gpg.key" \
    -O /usr/share/keyrings/apt.llvm.org.asc;
  echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/apt.llvm.org.asc] http://apt.llvm.org/${distribution}/ llvm-toolchain-${distribution}-${llvm_version} main" | \
    tee "/etc/apt/sources.list.d/llvm-${llvm_version}.list" > /dev/null;
  apt update;
  apt -y --no-install-recommends install ${pkg};
  return ${?};
}

pluto_apt() {
  local script_dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")";
  export DEBIAN_FRONTEND="noninteractive";
  . "${script_dir}/build-env.sh";
  apt update;
  apt -y dist-upgrade;
  apt -y --no-install-recommends install \
    lsb-release ca-certificates wget file patch git \
    make cmake ninja-build binutils-gold \
    gcc g++ gcc-multilib g++-multilib \
    autoconf automake libtool pkg-config flex bison texinfo \
    python3 python-is-python3 \
    zlib1g-dev libyaml-dev libgmp-dev glpk-utils libglpk-dev \
    nvidia-cuda-toolkit vim;
  #ccache
  llvm_install;
  return ${?};
}

pluto_apt;
