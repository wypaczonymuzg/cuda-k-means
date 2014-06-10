################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/mainLoop.cpp 

CU_SRCS += \
../src/simulation.cu 

CU_DEPS += \
./src/simulation.d 

OBJS += \
./src/mainLoop.o \
./src/simulation.o 

CPP_DEPS += \
./src/mainLoop.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc --compile -G -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


