#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include <vector>
#include <cmath>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN 
#define DTYPE_IN int16_t //std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT int16_t //std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

// ----------------------------------------------------------------------------
// Verify results (specific to our design example)
// ----------------------------------------------------------------------------
namespace test_utils{
    template <typename T>
    bool eq(T a, T b, T tolerance){
        return std::abs(a - b) <= tolerance;
    }
}

template <typename TC>
int verify(
    const std::vector<std::vector<TC>>& C,
    int verbosity = 1,
    float tolerance = 0.00390635f
){
    int errors = 0;
    size_t N = C.size(); 
    size_t M = C[0].size();

    for(size_t i = 0; i < N; ++i){
        for(size_t j = 0; j < M; ++j){
            float c_val = static_cast<DTYPE_ACC>(C[i][j]);
            float ref = static_cast<DTYPE_ACC>((i*M)+(2*j));
            if(std::abs(ref - c_val) > tolerance){
                std::cout << "Error at C[" << i << "][" << j << "]: "
                          << C[i][j] << "!=" << ref
                          << " from dot i: " << i << " j: " << j << "\n";
                errors++;
            } 
            else if(verbosity >= 0){
                std::cout << "Correct output C[" << i << "][" << j << "]: "
                          << C[i][j] << " == " << ref << "\n";
            }
        }
    }
    return errors;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]){

    // ------------------------------------------------------
    // Parse program arguments
    // ------------------------------------------------------
    cxxopts::Options options("Element wise matrix addition test");
    cxxopts::ParseResult vm;
    matmul_common::add_default_options(options);

    matmul_common::parse_options(argc, argv, options, vm);
    int verbosity = vm["verbosity"].as<int>();
    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();
    int trace_size = vm["trace_sz"].as<int>();

    // ------------------------------------------------------
    // Configure this to match your design's buffer size
    // ------------------------------------------------------
    int M = vm["rows"].as<int>();
    int N = vm["columns"].as<int>();

    size_t INA_SIZE = N*M * sizeof(A_DATATYPE); // X
    size_t INB_SIZE = M *  sizeof(B_DATATYPE); // H
    size_t OUTC_SIZE = N*M * sizeof(C_DATATYPE);

    size_t OUT_SIZE = OUTC_SIZE + trace_size;

    srand(time(NULL));
    
    // Load instruction sequence
    std::vector<uint32_t> instr_v = 
        test_utils::load_instr_binary(vm["instr"].as<std::string>());
    if (verbosity >= 1)
        std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    // ------------------------------------------------------
    // Get device, load the xclbin & kernel and register them
    // ------------------------------------------------------
    // Get a device handle
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    // Load the xclbin
    if (verbosity >= 1)
        std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
    
    // Load the kernel
    if (verbosity >= 1)
        std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
    std::string Node = vm["kernel"].as<std::string>();

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                [Node, verbosity](xrt::xclbin::kernel &k) {
                                    auto name = k.get_name();
                                    if (verbosity >= 1){
                                        std::cout << "Name: " << name << std::endl;
                                    }
                                    return name.rfind(Node, 0) == 0;
                                });
    auto kernelName = xkernel.get_name();

    // Register xclbin
    if(verbosity >= 1)
        std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
                  << "\n";
    device.register_xclbin(xclbin);

    // Get a hardware context
    if (verbosity >= 1)
        std::cout << "Getting hardware context.\n";
    xrt::hw_context context(device, xclbin.get_uuid());

    // Get a kernel handle
    if (verbosity >= 1)
        std::cout << "Getting handle to kernel:" << kernelName << "\n";
    auto kernel = xrt::kernel(context, kernelName);

    // ------------------------------------------------------
    // Initialize input/ output buffer sizes and sync them
    // ------------------------------------------------------
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_ina = 
        xrt::bo(device, INA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_inb = 
        xrt::bo(device, INB_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    // Assumes trace will only be added to outY - not sure what this means
    auto bo_outc = 
        xrt::bo(device, OUTC_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    
    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    // Initialize instruction buffer
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    // Initialize inX buffer
    A_DATATYPE *bufInA = bo_ina.map<A_DATATYPE *>();
    std::vector<std::vector<A_DATATYPE>> AVec(N, std::vector<A_DATATYPE>(M));
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            AVec[i][j] = (DTYPE_IN)((i*M)+j);
        }
    }

    // Flatten to buffer
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            size_t flat_ind = (i * M) + j;
            bufInA[flat_ind] = AVec[i][j];
        }
    }

    /***** For add and store testing
    // Initialize inW buffer
    B_DATATYPE *bufInB = bo_inb.map<B_DATATYPE *>();
    std::vector<std::vector<B_DATATYPE>> BVec(N, std::vector<B_DATATYPE>(M));
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            BVec[i][j] = (DTYPE_IN)((i*M) + j);
        }
    }

    // Flatten to buffer
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            size_t flat_ind = (i * M) + j;
            bufInB[flat_ind] = BVec[i][j];
        }
    }
    ********/ 
    
    // Initialize inW buffer
    B_DATATYPE *bufInB = bo_inb.map<B_DATATYPE *>();
    std::vector<B_DATATYPE> BVec(M);
    for(int j = 0; j < M; ++j){
        BVec[j] = (DTYPE_IN)(j);
    }
    
    // Flatten to buffer
    for(int j = 0; j < M; ++j){
        bufInB[j] = BVec[j];
    }
    

    // Initialize outY buffer
    char *bufOutC = bo_outc.map<char *>();
    memset(bufOutC, 0, OUTC_SIZE);

    // Sync buffers to update input buffer values
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ina.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ------------------------------------------------------
    // Initialize run configs
    // ------------------------------------------------------
    unsigned num_iter = n_iterations + n_warmup_iterations;
    float npu_time_total = 0;
    float npu_time_min = 9999999;
    float npu_time_max = 0;

    int errors = 0;

    // ------------------------------------------------------
    // Main run loop
    // ------------------------------------------------------
    for (unsigned iter = 0; iter < num_iter; iter++){
        if (verbosity >= 1){
            std::cout << "Running Kernel.\n";
        }

        // Run kernel
        if (verbosity >= 1)
            std::cout << "Running Kernel.\n";
        auto start = std::chrono::high_resolution_clock::now();
        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_ina, bo_inb, bo_outc);
        run.wait();
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Sync output \n"; 
        bo_outc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        if(iter < n_warmup_iterations){
            /* Warmup iterations do not count towards average runtime. */
            continue;
        }
       
        std::cout << "Reading output \n"; 
        // Copy output results and verify they are correct
        C_DATATYPE* typedBuf = reinterpret_cast<C_DATATYPE*>(bufOutC);
        // Create a 2D vector to store the result
        std::vector<std::vector<C_DATATYPE>> CVec(N, std::vector<C_DATATYPE>(M));
        // Fill CVec with the flattened output
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                size_t flat_index = i * M + j;  // Row-major layout
                CVec[i][j] = typedBuf[flat_index];
            }
        }

        if(do_verify){
            if(verbosity >= 1){
                std::cout << "Verifying results ..." << std::endl;
            }
            auto vstart = std::chrono::system_clock::now();
            errors = verify(CVec);
            auto vstop = std::chrono::system_clock::now();
            float vtime = 
                std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
                    .count();
            if (verbosity >= 1){
                std::cout << "Verify time: " << vtime << "secs." << std::endl;
            }
        }
        else{
            if(verbosity >= 1){
                std::cout << "WARNING: results not verified." << std::endl;
            }
        }

        // Write trace values if trace_size > 0
        if(trace_size > 0){
            test_utils::write_out_trace(((char *)bufOutC) + OUTC_SIZE, trace_size, 
                                      vm["trace_file"].as<std::string>());
        }

        // Accumulate run times
        float npu_time =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();

        npu_time_total += npu_time;
        npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
        npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
    }

    // ------------------------------------------------------
    // Display results
    // ------------------------------------------------------

    if(!errors){
        std::cout << "\nPass!\n\n";
        return 0;
    }
    else{
        std::cout << "\nError count: " << errors << "\n\n";
        std::cout << "\nFailed.\n\n";
        return 1;
    }
}
