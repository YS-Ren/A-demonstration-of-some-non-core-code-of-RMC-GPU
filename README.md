This project is a part of the RMC-GPU version. 

The program RMC (Reactor Monte Carlo code) was independently developed by the Institute of Reactor Engineering Computational Analysis Laboratory (REAL) of the Nuclear Energy Science and Engineering Management Department of the Engineering Physics Department of Tsinghua University. It is a three-dimensional particle transport Monte Carlo program used for reactor calculation analysis. 

Recently, we have attempted to transfer the neutron criticality function of RMC to the GPU side, and have made multiple algorithm adjustments based on the GPU architecture. 

Since RMC is protected by copyright, this project does not contain any original code of RMC. Only the non-core code added in the RMC-GPU version and used specifically for the review of the journal is included. The theories and methods used have been elaborated in the article "Research on Advanced GPU-Based Monte Carlo Neutron Transport Methods". The part presented in this project is an incomplete implementation of the scheme in the article and is provided for reference only. 

The code presented in this project is highly coupled with the original RMC, so it is not yet compatible for standalone operation. 

Among them, GPUBaseFunc.h, GPUContainer.h, GPUErrorCtrl.h, GPUException.h, GPUMemory.h, and GPUPrint.h are the basic tools of RMC for the GPU version. These header files jointly form the framework of the RMC-GPU version and possess high versatility. 

ParaTally_GPU.h and RNG_GPU.h are tools specifically designed for Monte Carlo simulations. 

The NeutronTracking_GPU.h and TrackHistory_GPU.cu files illustrate the basic control flow of the transport process, while concealing the specific implementation of the core functions.
