#include "Env.h"

#include <stdexcept>

clw::Env::Env() {
    int err;
    
    err = clGetPlatformIDs(1, &_platform, nullptr);
    if (err < 0) {
        throw std::runtime_error("Couldn't find an OpenCL platform (" + std::to_string(err) + ")");
    }

    err = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 1, &_device, NULL);
    if (err < 0) {
       throw std::runtime_error("Couldn't access any device (" + std::to_string(err) + ")");
    }

    _context = clCreateContext(nullptr, 1, &_device, nullptr, nullptr, &err);
    if (err < 0) {
       throw std::runtime_error("Couldn't create a context (" + std::to_string(err) + ")");
    }
}

clw::Env::~Env() {
    clReleaseContext(_context);
    clReleaseDevice(_device);
}

std::string clw::errorStr(cl_int error) {
    switch (error) {
    // runtime
    case 0: return "CL_SUCCESS (" + std::to_string(error) + ")";
    case -1: return "CL_DEVICE_NOT_FOUND (" + std::to_string(error) + ")";
    case -2: return "CL_DEVICE_NOT_AVAILABLE (" + std::to_string(error) + ")";
    case -3: return "CL_COMPILER_NOT_AVAILABLE (" + std::to_string(error) + ")";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE (" + std::to_string(error) + ")";
    case -5: return "CL_OUT_OF_RESOURCES (" + std::to_string(error) + ")";
    case -6: return "CL_OUT_OF_HOST_MEMORY (" + std::to_string(error) + ")";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE (" + std::to_string(error) + ")";
    case -8: return "CL_MEM_COPY_OVERLAP (" + std::to_string(error) + ")";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH (" + std::to_string(error) + ")";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED (" + std::to_string(error) + ")";
    case -11: return "CL_BUILD_PROGRAM_FAILURE (" + std::to_string(error) + ")";
    case -12: return "CL_MAP_FAILURE (" + std::to_string(error) + ")";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET (" + std::to_string(error) + ")";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST (" + std::to_string(error) + ")";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE (" + std::to_string(error) + ")";
    case -16: return "CL_LINKER_NOT_AVAILABLE (" + std::to_string(error) + ")";
    case -17: return "CL_LINK_PROGRAM_FAILURE (" + std::to_string(error) + ")";
    case -18: return "CL_DEVICE_PARTITION_FAILED (" + std::to_string(error) + ")";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE (" + std::to_string(error) + ")";
    
    // compile-time
    case -30: return "CL_INVALID_VALUE (" + std::to_string(error) + ")";
    case -31: return "CL_INVALID_DEVICE_TYPE (" + std::to_string(error) + ")";
    case -32: return "CL_INVALID_PLATFORM (" + std::to_string(error) + ")";
    case -33: return "CL_INVALID_DEVICE (" + std::to_string(error) + ")";
    case -34: return "CL_INVALID_CONTEXT (" + std::to_string(error) + ")";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES (" + std::to_string(error) + ")";
    case -36: return "CL_INVALID_COMMAND_QUEUE (" + std::to_string(error) + ")";
    case -37: return "CL_INVALID_HOST_PTR (" + std::to_string(error) + ")";
    case -38: return "CL_INVALID_MEM_OBJECT (" + std::to_string(error) + ")";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR (" + std::to_string(error) + ")";
    case -40: return "CL_INVALID_IMAGE_SIZE (" + std::to_string(error) + ")";
    case -41: return "CL_INVALID_SAMPLER (" + std::to_string(error) + ")";
    case -42: return "CL_INVALID_BINARY (" + std::to_string(error) + ")";
    case -43: return "CL_INVALID_BUILD_OPTIONS (" + std::to_string(error) + ")";
    case -44: return "CL_INVALID_PROGRAM (" + std::to_string(error) + ")";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE (" + std::to_string(error) + ")";
    case -46: return "CL_INVALID_KERNEL_NAME (" + std::to_string(error) + ")";
    case -47: return "CL_INVALID_KERNEL_DEFINITION (" + std::to_string(error) + ")";
    case -48: return "CL_INVALID_KERNEL (" + std::to_string(error) + ")";
    case -49: return "CL_INVALID_ARG_INDEX (" + std::to_string(error) + ")";
    case -50: return "CL_INVALID_ARG_VALUE (" + std::to_string(error) + ")";
    case -51: return "CL_INVALID_ARG_SIZE (" + std::to_string(error) + ")";
    case -52: return "CL_INVALID_KERNEL_ARGS (" + std::to_string(error) + ")";
    case -53: return "CL_INVALID_WORK_DIMENSION (" + std::to_string(error) + ")";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE (" + std::to_string(error) + ")";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE (" + std::to_string(error) + ")";
    case -56: return "CL_INVALID_GLOBAL_OFFSET (" + std::to_string(error) + ")";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST (" + std::to_string(error) + ")";
    case -58: return "CL_INVALID_EVENT (" + std::to_string(error) + ")";
    case -59: return "CL_INVALID_OPERATION (" + std::to_string(error) + ")";
    case -60: return "CL_INVALID_GL_OBJECT (" + std::to_string(error) + ")";
    case -61: return "CL_INVALID_BUFFER_SIZE (" + std::to_string(error) + ")";
    case -62: return "CL_INVALID_MIP_LEVEL (" + std::to_string(error) + ")";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE (" + std::to_string(error) + ")";
    case -64: return "CL_INVALID_PROPERTY (" + std::to_string(error) + ")";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR (" + std::to_string(error) + ")";
    case -66: return "CL_INVALID_COMPILER_OPTIONS (" + std::to_string(error) + ")";
    case -67: return "CL_INVALID_LINKER_OPTIONS (" + std::to_string(error) + ")";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT (" + std::to_string(error) + ")";
    case -69: return "CL_INVALID_PIPE_SIZE   (" + std::to_string(error) + ")";
    case -70: return "CL_INVALID_DEVICE_QUEUE (" + std::to_string(error) + ")";
    case -71: return "CL_INVALID_SPEC_ID (" + std::to_string(error) + ")";
    case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED (" + std::to_string(error) + ")";

    default: return "unknown error (" + std::to_string(error) + ")";
    }
}
