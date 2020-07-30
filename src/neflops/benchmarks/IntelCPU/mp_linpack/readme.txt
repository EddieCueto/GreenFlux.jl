================================================================================
Copyright 2001-2019 Intel Corporation.

This software and the related  documents are Intel  copyrighted  materials,  and
your use  of them  is governed  by the  express  license under  which  they were
provided to you (License).  Unless the  License provides otherwise,  you may not
use,  modify, copy,  publish, distribute,  disclose or transmit this software or
the related documents without Intel's prior written permission.

This software and the  related documents are provided as is,  with no express or
implied warranties, other than those that are expressly stated in the License.
================================================================================

======================================================================
 -- High Performance Computing Linpack Benchmark (HPL)                
    HPL - 2.3 - December 2, 2018                        
    Antoine P. Petitet                                                
    University of Tennessee, Knoxville                                
    Innovative Computing Laboratory                                 
    (C) Copyright 2000-2008 All Rights Reserved                       
                                                                      
 -- Copyright notice and Licensing terms:                             
                                                                      
 Redistribution  and  use in  source and binary forms, with or without
 modification, are  permitted provided  that the following  conditions
 are met:                                                             
                                                                      
 1. Redistributions  of  source  code  must retain the above copyright
 notice, this list of conditions and the following disclaimer.        
                                                                      
 2. Redistributions in binary form must reproduce  the above copyright
 notice, this list of conditions,  and the following disclaimer in the
 documentation and/or other materials provided with the distribution. 
                                                                      
 3. All  advertising  materials  mentioning  features  or  use of this
 software must display the following acknowledgement:                 
 This  product  includes  software  developed  at  the  University  of
 Tennessee, Knoxville, Innovative Computing Laboratory.             
                                                                      
 4. The name of the  University,  the name of the  Laboratory,  or the
 names  of  its  contributors  may  not  be used to endorse or promote
 products  derived   from   this  software  without  specific  written
 permission.                                                          
                                                                      
 -- Disclaimer:                                                       
                                                                      
 THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
======================================================================

===============================================================================

       Intel(R) Distribution for LINPACK* Benchmark

===============================================================================

Package Contents 
----------------

This package contains Intel(R) Distribution for LINPACK Benchmark.
It is optimized for Intel(R) Xeon(R) and Intel(R) Xeon Phi(TM) processors. 
First generation of Intel(R) Xeon Phi(TM) Product Family (codename Knights
Corner) is not supported.

Pre-built binaries linked with Intel(R) MPI are included in this package. 
In addition, Intel Distribution for LINPACK Benchmark binary linked with
a third party MPI implementation can be created using the Intel(R) MKL MPI
wrappers. 

The package contains the following files:

COPYRIGHT                 : Original Netlib HPL copyright document

readme.txt                : This document

runme_intel64_dynamic     : Run script for dynamically linked MPI 

runme_intel64_static      : Run script for statically linked MPI 

runme_intel64_prv         : Common script used by above two scripts 

xhpl_intel64_dynamic      : Intel Distribution for LINPACK Benchmark binary
                            dynamically linked with Intel(R) MPI

xhpl_intel64_static       : Intel Distribution for LINPACK Benchmark binary
                            statically linked with Intel(R) MPI

HPL.dat                   : HPL configuration file

build.sh                  : Build script for creating Intel Distribution for
                            LINPACK Benchmark binary linked with third party
                            MPI implementation 

HPL_main.c                : Source code required to build Intel Distribution
                            for LINPACK Benchmark binary with third party MPI
                            implementation.

libhpl_intel64.a          : Library file required to build Intel Distribution
                            for LINPACK Benchmark binary with third party MPI
                            implementation. 

Blocking size (NB) recommendation
---------------------------------

Recommended blocking sizes (NB in HPL.dat) are listed below for various Intel(R) 
architectures:

Intel(R) Xeon(R) Processor X56*/E56*/E7-*/E7*/X7*                             : 256
Intel(R) Xeon(R) Processor E26*/E26* v2                                       : 256
Intel(R) Xeon(R) Processor E26* v3/E26* v4                                    : 192
Intel(R) Core(TM) i3/5/7-6* Processor                                         : 192
Intel(R) Xeon Phi(TM) Processor 72*                                           : 336
Intel(R) Xeon(R) Scalabile Processors                                         : 384

Building Intel Distribution for LINPACK Benchmark for third party MPI
-------------------------------------------------------------

After setting MPI environment, please run build.sh script in this directory.
"xhpl" binary will be built after executing "./build.sh" script.

Building Intel Distribution for LINPACK Benchmark from source code
----------------------------------------------

Netlib HPL source code can be obtained from

    http://www.netlib.org/benchmark/hpl/

After extracting source code, 

1. Copy makefile

     $> cp setup/Make.Linux_Intel64 .

2. Edit Make.Linux_intel64 appropriately

3. Build HPL binary

     $> make arch=Linux_Intel64

4. Binary will be located in bin/Linux_Intel64 directory

---
*Other names and brands may be claimed as the property of others.
