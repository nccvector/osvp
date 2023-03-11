//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "Shared.h"
#include <cuda/helpers.h>


extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    const uint3 launch_index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch launch_index to a screen location and create a ray from 
    // the camera location through the screen
    float3 ray_origin, ray_direction;

    ray_origin.x = launch_index.x;
    ray_origin.y = launch_index.y;
    ray_origin.z = 0.f;

    // Normalized ray direction
    float sum = launch_index.x + launch_index.y + 1.f;
    ray_direction.x = launch_index.x / sum;
    ray_direction.y = launch_index.y / sum;
    ray_direction.z = 1.f / sum;
 
    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,   // Min intersection distance
        1e16f,  // Max intersection distance
        0.0f,   // ray-time -- used for motion blur
        OptixVisibilityMask( 255 ), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,      // SBT offset -- See SBT discussion
        0,      // SBT stride -- See SBT discussion 
        0,      // missSBTIndex -- See SBT discussion
        p0, p1, p2 ); // These 32b values are the ray payload
 
    // Our results were packed into opaque 32b registers
    float3 result;
    result.x = int_as_float( p0 );
    result.y = int_as_float( p1 );
    result.z = int_as_float( p2 );
 
    // Record results in our output raster
    params.image[launch_index.y * params.image_width + launch_index.x] = make_color( result );
}
