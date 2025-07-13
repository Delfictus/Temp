# ARES Configuration File
include(CMakeFindDependencyMacro)

find_dependency(Threads)

include("${CMAKE_CURRENT_LIST_DIR}/ARESTargets.cmake")