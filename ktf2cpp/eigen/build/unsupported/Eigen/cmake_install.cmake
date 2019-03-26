# Install script for directory: /home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/AdolcForward"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/AlignedVector3"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/ArpackSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/AutoDiff"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/BVH"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/EulerAngles"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/FFT"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/IterativeSolvers"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/KroneckerProduct"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/MatrixFunctions"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/MoreVectorization"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/MPRealSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/NonLinearOptimization"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/NumericalDiff"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/OpenGLSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/Polynomials"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/Skyline"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/SparseExtra"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/SpecialFunctions"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

