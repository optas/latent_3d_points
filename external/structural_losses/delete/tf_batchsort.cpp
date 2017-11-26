#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
REGISTER_OP("BatchSort")
	.Input("inp: float32")
	.Output("out: float32")
	.Output("outi: int32");
REGISTER_OP("BatchSortGrad")
	.Input("grad_out: float32")
	.Input("outi: int32")
	.Output("grad_inp: float32");
#include <algorithm>
#include <vector>
using namespace tensorflow;
class BatchSortOp : public OpKernel{
	public:
		explicit BatchSortOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& inp_tensor=context->input(0);
			OP_REQUIRES(context,inp_tensor.dims()==2,errors::InvalidArgument("BatchSort requires inp be of shape (batch,#numbers)"));
			int b=inp_tensor.shape().dim_size(0);
			int n=inp_tensor.shape().dim_size(1);
			auto inp_flat=inp_tensor.flat<float>();
			const float * inp=&inp_flat(0);
			Tensor * out_tensor=NULL;
			Tensor * outi_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&out_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&outi_tensor));
			auto out_flat=out_tensor->flat<float>();
			float * out=&(out_flat(0));
			auto outi_flat=outi_tensor->flat<int>();
			int * outi=&(outi_flat(0));
			std::vector<std::pair<float,int> > pairs(n);
			for (int i=0;i<b;i++){
				for (int j=0;j<n;j++)
					pairs[j]=std::make_pair(inp[i*n+j],j);
				std::sort(pairs.begin(),pairs.end());
				for (int j=0;j<n;j++){
					out[i*n+j]=pairs[j].first;
					outi[i*n+j]=pairs[j].second;
				}
			}
		}
};
REGISTER_KERNEL_BUILDER(Name("BatchSort").Device(DEVICE_CPU), BatchSortOp);
void BatchSortKernelLauncher(int b,int n,const float * inp,float * out,int * out_i);
class BatchSortGpuOp : public OpKernel{
	public:
		explicit BatchSortGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& inp_tensor=context->input(0);
			OP_REQUIRES(context,inp_tensor.dims()==2,errors::InvalidArgument("BatchSort requires inp be of shape (batch,#numbers)"));
			int b=inp_tensor.shape().dim_size(0);
			int n=inp_tensor.shape().dim_size(1);
			auto inp_flat=inp_tensor.flat<float>();
			const float * inp=&inp_flat(0);
			Tensor * out_tensor=NULL;
			Tensor * outi_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&out_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&outi_tensor));
			auto out_flat=out_tensor->flat<float>();
			float * out=&(out_flat(0));
			auto outi_flat=outi_tensor->flat<int>();
			int * outi=&(outi_flat(0));
			BatchSortKernelLauncher(b,n,inp,out,outi);
		}
};
REGISTER_KERNEL_BUILDER(Name("BatchSort").Device(DEVICE_GPU), BatchSortGpuOp);
class BatchSortGradOp : public OpKernel{
	public:
		explicit BatchSortGradOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& grad_out_tensor=context->input(0);
			const Tensor& outi_tensor=context->input(1);
			OP_REQUIRES(context,grad_out_tensor.dims()==2,errors::InvalidArgument("BatchSortGrad requires grad_out be of shape (batch,#numbers)"));
			int b=grad_out_tensor.shape().dim_size(0);
			int n=grad_out_tensor.shape().dim_size(1);
			OP_REQUIRES(context,outi_tensor.dims()==2,errors::InvalidArgument("BatchSortGrad requires outi be of shape (batch,#numbers)"));
			OP_REQUIRES(context,outi_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("BatchSortGrad requires outi and grad_out have the same shape"));
			auto grad_out_flat=grad_out_tensor.flat<float>();
			const float * grad_out=&grad_out_flat(0);
			auto outi_flat=outi_tensor.flat<int>();
			const int * outi=&outi_flat(0);

			Tensor * grad_inp_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&grad_inp_tensor));
			auto grad_inp_flat=grad_inp_tensor->flat<float>();
			float * grad_inp=&grad_inp_flat(0);
			for (int i=0;i<b;i++){
				for (int j=0;j<n;j++){
					int idx=outi[i*n+j];
					grad_inp[i*n+idx]=grad_out[i*n+j];
				}
			}
		}
};
REGISTER_KERNEL_BUILDER(Name("BatchSortGrad").Device(DEVICE_CPU), BatchSortGradOp);
void BatchSortGradKernelLauncher(int b,int n,const float * grad_out,const int * out_i,float * grad_inp);
class BatchSortGradGpuOp : public OpKernel{
	public:
		explicit BatchSortGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& grad_out_tensor=context->input(0);
			const Tensor& outi_tensor=context->input(1);
			OP_REQUIRES(context,grad_out_tensor.dims()==2,errors::InvalidArgument("BatchSortGrad requires grad_out be of shape (batch,#numbers)"));
			int b=grad_out_tensor.shape().dim_size(0);
			int n=grad_out_tensor.shape().dim_size(1);
			OP_REQUIRES(context,outi_tensor.dims()==2,errors::InvalidArgument("BatchSortGrad requires outi be of shape (batch,#numbers)"));
			OP_REQUIRES(context,outi_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("BatchSortGrad requires outi and grad_out have the same shape"));
			auto grad_out_flat=grad_out_tensor.flat<float>();
			const float * grad_out=&grad_out_flat(0);
			auto outi_flat=outi_tensor.flat<int>();
			const int * outi=&outi_flat(0);

			Tensor * grad_inp_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&grad_inp_tensor));
			auto grad_inp_flat=grad_inp_tensor->flat<float>();
			float * grad_inp=&grad_inp_flat(0);
			BatchSortGradKernelLauncher(b,n,grad_out,outi,grad_inp);
		}
};
REGISTER_KERNEL_BUILDER(Name("BatchSortGrad").Device(DEVICE_GPU), BatchSortGradGpuOp);
