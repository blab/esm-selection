#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::BTF" for configuration "Release"
set_property(TARGET SuiteSparse::BTF APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::BTF PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libbtf.2.3.2.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libbtf.2.dylib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::BTF )
list(APPEND _cmake_import_check_files_for_SuiteSparse::BTF "${_IMPORT_PREFIX}/lib/libbtf.2.3.2.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
