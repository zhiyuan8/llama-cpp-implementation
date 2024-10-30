# whisper

# Find the Threads package
find_package(Threads REQUIRED)

# build whisper-utils
set(WHISPER_LIB qwen2-whisper-utils)  # Changed from whisper-utils
add_library(${WHISPER_LIB} OBJECT whisper.cpp grammar-parser.cpp)
target_link_libraries(${WHISPER_LIB} PRIVATE ggml common Threads::Threads)

# build the qwen2-audio library
set(QWEN2_AUDIO_LIB hf-qwen2-audio)
add_library(${QWEN2_AUDIO_LIB} OBJECT
            qwen2audio.cpp
            qwen2audio.h
            audio-projector.cpp
            audio-projector.h
            )
target_link_libraries(${QWEN2_AUDIO_LIB} PRIVATE ggml common Threads::Threads ${WHISPER_LIB})

# If BUILD_SHARED_LIBS is ON, also build a shared library
if(BUILD_SHARED_LIBS)
    message(STATUS "Building qwen2-audio shared libraries")

    set_target_properties(${WHISPER_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${QWEN2_AUDIO_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(${QWEN2_AUDIO_LIB}_shared SHARED $<TARGET_OBJECTS:${QWEN2_AUDIO_LIB}>)
    target_link_libraries(${QWEN2_AUDIO_LIB}_shared PRIVATE ggml common ${WHISPER_LIB})
    set_target_properties(${QWEN2_AUDIO_LIB}_shared PROPERTIES
                            PUBLIC_HEADER qwen2audio.h
                            POSITION_INDEPENDENT_CODE ON
    )

    target_compile_definitions(${QWEN2_AUDIO_LIB}_shared PRIVATE QWEN2_AUDIO_SHARED WHISPER_SHARED)

    if(MSVC)
        set_target_properties(${QWEN2_AUDIO_LIB}_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()


# # Add debug option
# option(NEXA_DEBUG "Enable Nexa debug output" ON)
# option(WHISPER_DEBUG "Enable Whisper debug output" ON)

# # If debug is enabled, add the definition
# if(NEXA_DEBUG)
#     target_compile_definitions(${TARGET} PRIVATE NEXA_DEBUG WHISPER_DEBUG)
# endif()
