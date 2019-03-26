# Install script for directory: /home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Cholesky"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/CholmodSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Core"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Dense"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Eigen"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Eigenvalues"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Geometry"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Householder"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/IterativeLinearSolvers"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Jacobi"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/KLUSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/LU"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/MetisSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/OrderingMethods"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/PaStiXSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/PardisoSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/QR"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/QtAlignedMalloc"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/SPQRSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/SVD"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/Sparse"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/SparseCholesky"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/SparseCore"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/SparseLU"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/SparseQR"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/StdDeque"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/StdList"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/StdVector"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/SuperLUSupport"
    "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/UmfPackSupport"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "/home/roryconlin/GoogleDrive/SCHOOL/Princeton/PPPL/eigen/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

