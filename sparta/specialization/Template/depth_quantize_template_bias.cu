__global__ void DepthConvForwardNHWC(
    float*  input0, float* input1, float* input2, float * input3, float *input4, float * output0) {

    // launch config
    uint8_t * bottom_data = reinterpret_cast<uint8_t*>(input0);
    uint8_t * weight = reinterpret_cast<uint8_t*>(input1);
    const int * bias =  reinterpret_cast<int*>(input2);
    const int integer = (int)(*input3);
    const int shift = (int)(*input4);
    // const int nthreads = 1605632;
    const int channels = CHANNEL_VALUE;
    const int height = IN_HEIGHT_VALUE;
    const int width = IN_WIDTH_VALUE;
    const int conved_height = OUT_HEIGHT_VALUE;
    const int conved_width = OUT_WIDTH_VALUE;
    const int batchsize = BATCHSIZE_VALUE;
    int nthreads = conved_height * conved_width * channels * batchsize;
    const int kernel_h = KERNEL_H_VALUE;
    const int kernel_w = KERNEL_W_VALUE;
    const int stride_h = STRIDE_H_VALUE;
    const int stride_w = STRIDE_W_VALUE;
    const int pad_h = PAD_H_VALUE;
    const int pad_w = PAD_W_VALUE;
    uint8_t * top_data = reinterpret_cast<uint8_t*>(output0);


#pragma unroll
        for (int index = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4; \
            index < (nthreads); \
            index += blockDim.x * gridDim.x * 4){
 
    // get nhwc
    const int c = index % channels;
    const int pw = (index / channels) % conved_width;
    const int ph = (index / channels / conved_width) % conved_height;
    const int n = index / channels / conved_width / conved_height;
 
    // get range of height and width
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
 
//      const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
 
    int aveval = 0;
    // uint8_t* bottom_slice = bottom_data + (n * channels + c) * height * width;
    uint8_t* weight_slice =
    weight + c * kernel_h * kernel_w;
 
    int khstart=hend<kernel_h?kernel_h-hend:0;
    int kwstart=wend<kernel_w?kernel_w-wend:0;
 
    register uint8_t bottom_reg[4] = {0};
    register uint8_t weight_reg[4] = {0};
 
#pragma unroll
    for(int h = hstart; h < hend; ++h) {
    #pragma unroll
        for(int w = wstart; w < wend; ++w) {
            bottom_reg[w-wstart] = 
                bottom_data[n * height * width * channels+ h * width * channels + w * channels + c];
            weight_reg[w-wstart] = weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
        }
        int pack_val1 = reinterpret_cast<int*>(&(bottom_reg[0]))[0];
        int pack_val2 = reinterpret_cast<int*>(&(weight_reg[0]))[0];
        aveval = __dp4a(pack_val1, pack_val2, aveval);
    }
 

    aveval+=bias[c];

    aveval = ((aveval * integer) >> shift);
    top_data[index] = aveval > 0 ? (uint8_t)aveval:0;
}
}