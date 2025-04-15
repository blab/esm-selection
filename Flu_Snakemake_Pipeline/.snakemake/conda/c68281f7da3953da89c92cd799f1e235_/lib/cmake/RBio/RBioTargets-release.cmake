#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::RBio" for configuration "Release"
set_property(TARGET SuiteSparse::RBio APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::RBio PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librbio.4.3.4.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/librbio.4.dylib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::RBio )
list(APPEND _cmake_import_check_files_for_SuiteSparse::RBio "${_IMPORT_PREFIX}/lib/librbio.4.3.4.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
