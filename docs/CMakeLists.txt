find_package(Doxygen QUIET)

if (DOXYGEN_FOUND)

    set(DOXYGEN_IN   "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile")
    set(DOXYGEN_OUT  "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

    file(READ   "${DOXYGEN_IN}" _doxy_cfg)
    string(REGEX REPLACE
        "^(OUTPUT_DIRECTORY[ \t]*=).*"
        "\\1 ${CMAKE_CURRENT_BINARY_DIR}/html"
        _doxy_cfg
        "${_doxy_cfg}"
    )
    file(WRITE  "${DOXYGEN_OUT}" "${_doxy_cfg}")

    add_custom_target(doc_doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYGEN_OUT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else()
    message(STATUS "Doxygen not found, 'doc_doxygen' target will not be available.")
endif()
