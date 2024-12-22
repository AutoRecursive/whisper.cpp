add_executable(whisper-assistant
    assistant.cpp
    )

find_package(CURL REQUIRED)

target_link_libraries(whisper-assistant PRIVATE
    whisper
    common
    common-sdl
    ${CMAKE_THREAD_LIBS_INIT}
    CURL::libcurl
    )

target_include_directories(whisper-assistant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CURL_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ) 