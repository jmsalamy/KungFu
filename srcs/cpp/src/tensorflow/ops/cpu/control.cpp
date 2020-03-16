#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(ResizeCluster)
    .Input("checkpoint: string")
    .Input("new_cluster_size: int32")
    // indicates if cluster is changed
    .Output("changed: bool")
    // indicates if self is still in the new cluster
    .Output("keep: bool")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
    });

class ResizeCluster : public OpKernel
{
    bool debug_;

  public:
    ResizeCluster(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("debug", &debug_);
    }

    void Compute(OpKernelContext *context) override
    {
        const std::string &chpt = context->input(0).scalar<std::string>()();
        const int32_t new_size  = context->input(1).scalar<int32_t>()();
        if (debug_) {
            LOG(WARNING) << "ResizeCluster::Compute called with chpt: " << chpt
                         << " new size: " << new_size;
        }
        Tensor *changed = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, MakeTensorShape(), &changed));
        Tensor *keep = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, MakeTensorShape(), &keep));
        _kungfu_world->ResizeCluster(chpt.c_str(), new_size,
                                     changed->scalar<bool>().data(),
                                     keep->scalar<bool>().data());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResizeCluster, DEVICE_CPU);

REGISTER_KUNGFU_OP(ReshapeStrategy)
    //Global step information for iteration level tracking 
    .Input("global_step: int32")
    // indicates if strategy is changed
    .Output("changed: bool")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class ReshapeStrategy : public OpKernel
{
    bool debug_;

  public:
    ReshapeStrategy(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("debug", &debug_);
    }

    void Compute(OpKernelContext *context) override
    {
        const int32_t global_step  = context->input(0).scalar<int32_t>()();
        Tensor *changed = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, MakeTensorShape(), &changed));
        _kungfu_world->ReshapeStrategy(global_step, changed->scalar<bool>().data());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ReshapeStrategy, DEVICE_CPU);
}