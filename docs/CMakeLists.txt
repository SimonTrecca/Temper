find_package(Doxygen QUIET)

if (DOXYGEN_FOUND)
  set(DOXYFILE_IN   "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
  set(DOXYFILE_OUT  "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

  configure_file(
    ${DOXYFILE_IN}
    ${DOXYFILE_OUT}
    @ONLY
  )

  add_custom_target(doc_doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )

  find_package(Python COMPONENTS Interpreter QUIET)

  if (Python_EXECUTABLE)
    # Path where Doxygen will place XML according to the configured Doxyfile.
    # If your Doxyfile sets OUTPUT_DIRECTORY = @CMAKE_CURRENT_BINARY_DIR@/doxygen
    # then the XML will be at ${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml
    set(DOXYGEN_XML_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml")

    add_custom_target(doc_sphinx
      COMMAND ${Python_EXECUTABLE} -m sphinx -b html
              ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/source
              ${CMAKE_CURRENT_BINARY_DIR}/sphinx/html
              -D breathe_projects.Temper=${DOXYGEN_XML_DIR}
              -D breathe_default_project=Temper
      DEPENDS doc_doxygen
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Building Sphinx HTML documentation (requires sphinx in Python environment)"
      VERBATIM
    )
  else()
    message(STATUS "Python interpreter not found: doc_sphinx target will not be available")
  endif()

else()
  message(STATUS "Doxygen not found: target will not be available")
endif()
