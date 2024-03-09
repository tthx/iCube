#!/bin/bash
set -euo pipefail;

setup_poly() {
  local script_dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")";
  "${script_dir}/poly-apt.sh";
  "${script_dir}/build-clang.sh";
  "${script_dir}/build-isl.sh";
  "${script_dir}/build-pet.sh";
  return ${?};
}

setup_poly;
