#!/bin/bash
set -euo pipefail;

backup() {
  local src_dir="${1}";
  if [ -f "${src_dir}/reference/CMakeLists.txt" ];
  then
    mv -f "${src_dir}/reference/CMakeLists.txt" \
      "${src_dir}/reference/CMakeLists.txt.orig";
  fi
  return 0;
}

restore() {
  local src_dir="${1}";
  if [ -f "${src_dir}/reference/CMakeLists.txt.orig" ];
  then
    mv -f "${src_dir}/reference/CMakeLists.txt.orig" \
      "${src_dir}/reference/CMakeLists.txt";
  fi
  return 0;
}

apply_patches() {
  local src_dir="${1}";
  local new_dir="${src_home}/iCube/ginkgo/${ginkgo_branch}";
  cp -f "${new_dir}/reference/CMakeLists.txt" \
    "${src_dir}/reference/CMakeLists.txt";
  return 0;
}

build_ginkgo() {
  local script_dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")";
  . "${script_dir}/runtime-env.sh";
  local errmsg="ERROR: ${FUNCNAME[0]}:";
  local embedded="";
  if [ ${#} -gt 0 ];
  then
    if [ "${1}" == "embedded" ];
    then
      embedded="${1}";
      shift;
    fi
  else
    echo "Usage: ${BASH_SOURCE} [embedded] <cuda_arch> <${mpi_impl_list//\ /\|}> [gcc|llvm](default:gcc) [poly](default:OFF) [cxx_dialect](default:17) [openmp](default:OFF)";
    return 0;
  fi
  local cuda_arch="${1:?"${errmsg} Missing CUDA arch (e.g. 61, 86)"}";
  local mpi_impl="${2:?"${errmsg} Missing MPI implementation, supported are: [${mpi_impl_list//\ /,\ }]"}";
  check_mpi_impl "${mpi_impl}";
  cuda_runtime_env;
  if [ -z "${embedded}" ];
  then
    hwloc_runtime_env;
    ucx_runtime_env;
    mpi_impl_runtime_env "${mpi_impl}";
  fi
  petsc_runtime_env "${embedded}" "${mpi_impl}";
  local type="${3:-${gcc_type}}";
  local poly="${4:-OFF}";
  local cc="/usr/bin/gcc";
  local cflags="${common_cflags}";
  local cxx="/usr/bin/g++";
  local cxxflags="${common_cxxflags}";
  local cxx_dialect="${5:-17}";
  local openmp="${6:-OFF}";
  local cudac="$(which nvcc)";
  local cudaflags="-arch=sm_${cuda_arch}";
  local fortran="/usr/bin/gfortran";
  local fcflags="${cflags}";
  local ldflags="";
  local src_dir="${ginkgo_polyhedral_src_dir}";
  local branch="${ginkgo_polyhedral_branch}";
  local repo_url="${ginkgo_polyhedral_repo_url}";
  local prefix="${ginkgo_polyhedral_prefix}";
  local dev="ON";
  local version="${ginkgo_branch}";
  if [ "${type}" == "${llvm_type}" ];
  then
    llvm_runtime_env;
    cc="$(llvm-config --bindir)/clang";
    cflags+=" -Wno-unused-but-set-variable";
    cxx="$(llvm-config --bindir)/clang++";
    cxxflags+=" -Wno-unused-but-set-variable";
    cudac="${cxx}";
    cudaflags=" --cuda-gpu-arch=sm_${cuda_arch} -lcudart_static -ldl -lrt -pthread";
    if [ "${poly}" == "ON" ];
    then
      cflags+=" ${polly_cflags}";
      cxxflags+=" ${polly_cxxflags}";
      fflags+=" ${graphite_cflags}";
    fi
    type+="-${poly}-$(llvm-config --version)"
  else
    if [ "${poly}" == "ON" ];
    then
      cflags+=" ${graphite_cflags}";
      cxxflags+=" ${graphite_cxxflags}";
      fflags="${cflags}";
    fi
    type+="-${poly}-$(${cc} --version | awk '/^gcc/{print $4}')"
  fi
  if [ "${version}" != "${ginkgo_polyhedral_branch}" ];
  then
    src_dir="${ginkgo_src_dir}"
    branch="${ginkgo_branch}";
    repo_url="${ginkgo_repo_url}";
    prefix="${ginkgo_prefix}";
    dev="OFF";
  fi
  if [ ! -d "${src_dir}" ];
  then
    git clone \
      --depth=1 \
      --recursive \
      -b "${branch}" \
      "${repo_url}" \
      "${src_dir}";
  fi
  cd "${src_dir}";
  git pull --recurse-submodules;
  restore "${src_dir}";
  if [ "${type}" =~ "${llvm_type}" ];
  then
    backup "${src_dir}";
    apply_patches "${src_dir}";
  fi
  prefix+="${embedded:+/embedded}/${mpi_impl}";
  rm -rf "${prefix}" "./build";
  mkdir -p "./build";
  cd "./build";
  export CC="${cc}";
  export CFLAGS="${cflags}";
  export CXX="${cxx}";
  export CXXFLAGS="${cxxflags}";
  export CUDAC="${cudac}"
  export CUDACXX="${cudac}";
  export CUDAFLAGS="${cudaflags}";
  export LDFLAGS="${ldflags}";
  cmake -G Ninja .. \
    -DCMAKE_C_COMPILER="${cc}" \
    -DCMAKE_C_FLAGS="${cflags}" \
    -DCMAKE_CXX_COMPILER="${cxx}" \
    -DCMAKE_CXX_FLAGS="${cxxflags}" \
    -DCMAKE_CXX_STANDARD="${cxx_dialect}" \
    -DCMAKE_CUDA_COMPILER="${cudac}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DCMAKE_CUDA_FLAGS="${cudaflags}" \
    -DCUDA_STANDARD="${cxx_dialect}" \
    -DCMAKE_EXE_LINKER_FLAGS="${ldflags}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${ldflags}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${ldflags}" \
    -DCMAKE_INSTALL_PREFIX="${prefix}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGINKGO_BUILD_OMP="${openmp}" \
    -DGINKGO_BUILD_MPI=ON \
    -DGINKGO_BUILD_SYCL=OFF \
    -DGINKGO_BUILD_DPCPP=OFF \
    -DGINKGO_BUILD_HIP=OFF \
    -DGINKGO_BUILD_CUDA=ON \
    -DGINKGO_JACOBI_FULL_OPTIMIZATIONS=ON \
    -DGINKGO_MIXED_PRECISION=ON \
    -DGINKGO_BUILD_EXAMPLES=OFF \
    -DGINKGO_BUILD_TESTS=ON \
    -DGINKGO_BUILD_BENCHMARKS=ON \
    -DGINKGO_WITH_CCACHE=ON \
    -DGINKGO_DEVEL_TOOLS="${dev}";
  ninja -j $(nproc);
  ninja install;
  restore "${src_dir}";
  mpi_env_var;
  ninja -j $(nproc) test;
  return ${?};
}

build_ginkgo "${@}";
