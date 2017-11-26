#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
REGISTER_OP("ZeroGrad")
	.Input("inp: float32")
	.Output("out: float32");

#include <string.h>
#include <cuda_runtime.h>

using namespace tensorflow;

class ZeroGradOp: public OpKernel{
	public:
		explicit ZeroGradOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& inp_tensor=context->input(0);
			auto inp_flat=inp_tensor.flat<float>();
			size_t volume=4;
			for (int i=0;i<inp_tensor.dims();i++){
				volume*=inp_tensor.shape().dim_size(i);
			}
			const float * inp=&(inp_flat(0));
			Tensor * out_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,inp_tensor.shape(),&out_tensor));
			auto out_flat=out_tensor->flat<float>();
			float * out=&(out_flat(0));
			memcpy(out,inp,volume);
		}
};
REGISTER_KERNEL_BUILDER(Name("ZeroGrad").Device(DEVICE_CPU), ZeroGradOp);

class ZeroGradGpuOp: public OpKernel{
	public:
		explicit ZeroGradGpuOp(OpKernelConstruction * context): OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& inp_tensor=context->input(0);
			auto inp_flat=inp_tensor.flat<float>();
			size_t volume=4;
			for (int i=0;i<inp_tensor.dims();i++){
				volume*=inp_tensor.shape().dim_size(i);
			}
			const float * inp=&(inp_flat(0));
			Tensor * out_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,inp_tensor.shape(),&out_tensor));
			auto out_flat=out_tensor->flat<float>();
			float * out=&(out_flat(0));
			cudaMemcpy(out,inp,volume,cudaMemcpyDeviceToDevice);
		}
};
REGISTER_KERNEL_BUILDER(Name("ZeroGrad").Device(DEVICE_GPU), ZeroGradGpuOp);
