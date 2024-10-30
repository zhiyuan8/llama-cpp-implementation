# whisper

# Find the Threads package
find_package(Threads REQUIRED)

# build whisper-utils
set(WHISPER_LIB whisper-utils)
add_library(${WHISPER_LIB} OBJECT whisper.cpp grammar-parser.cpp)
target_link_libraries(${WHISPER_LIB} PRIVATE ggml common Threads::Threads)

# build the omni-audio-cli
set(OMNI_AUDIO_LIB hf-omni-audio-cli)
add_library(${OMNI_AUDIO_LIB} OBJECT
            omni.cpp
            omni.h
            audio-projector.cpp
            audio-projector.h
            )
target_link_libraries(${OMNI_AUDIO_LIB} PRIVATE ggml common Threads::Threads ${WHISPER_LIB})

# build the omni-cli
add_executable(hf-omni-cli omni-cli.cpp)
target_link_libraries(hf-omni-cli PRIVATE ggml common Threads::Threads ${WHISPER_LIB} ${OMNI_AUDIO_LIB})

# If BUILD_SHARED_LIBS is ON, also build a shared library
if(BUILD_SHARED_LIBS)
    message(STATUS "Building omni-audio shared libraries")

    set_target_properties(${WHISPER_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${OMNI_AUDIO_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(${OMNI_AUDIO_LIB}_shared SHARED $<TARGET_OBJECTS:${OMNI_AUDIO_LIB}>)
    target_link_libraries(${OMNI_AUDIO_LIB}_shared PRIVATE ggml common ${WHISPER_LIB})
    set_target_properties(${OMNI_AUDIO_LIB}_shared PROPERTIES
                            PUBLIC_HEADER omni.h
                            POSITION_INDEPENDENT_CODE ON
    )

    # Add OMNI_AUDIO_SHARED definition when building the shared library
    target_compile_definitions(${OMNI_AUDIO_LIB}_shared PRIVATE OMNI_AUDIO_SHARED WHISPER_SHARED)

    # Ensure all symbols are exported on Windows
    if(MSVC)
        set_target_properties(${OMNI_AUDIO_LIB}_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()