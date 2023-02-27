// These need to match for both HOST and DEVICE, hence the name shared
// These will be used to communicate data from HOST->DEVICE and DEVICE->HOST

// // This is a struct used to communicate launch parameters which are constant
// for all threads in a given optixLaunch call.
struct Params
{
    uchar4* image;
    unsigned int image_width;
};

struct RayGenData
{
    float r,g,b;
};
