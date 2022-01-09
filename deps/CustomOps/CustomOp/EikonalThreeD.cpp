#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "Eikonal3D.h"


REGISTER_OP("EikonalThreeD")
.Input("u0 : double")
.Input("f : double")
.Input("h : double")
.Input("m : int64")
.Input("n : int64")
.Input("l : int64")
.Input("tol : double")
.Input("verbose : bool")
.Output("u : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u0_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u0_shape));
        shape_inference::ShapeHandle f_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &f_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &h_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &n_shape));
        shape_inference::ShapeHandle l_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &l_shape));
        shape_inference::ShapeHandle tol_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &tol_shape));
        shape_inference::ShapeHandle verbose_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &verbose_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("EikonalThreeDGrad")
.Input("grad_u : double")
.Input("u : double")
.Input("u0 : double")
.Input("f : double")
.Input("h : double")
.Input("m : int64")
.Input("n : int64")
.Input("l : int64")
.Input("tol : double")
.Input("verbose : bool")
.Output("grad_u0 : double")
.Output("grad_f : double")
.Output("grad_h : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_l : int64")
.Output("grad_tol : double")
.Output("grad_verbose : bool");

/*-------------------------------------------------------------------------------------*/

class EikonalThreeDOp : public OpKernel {
private:
  
public:
  explicit EikonalThreeDOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(8, context->num_inputs());
    
    
    const Tensor& u0 = context->input(0);
    const Tensor& f = context->input(1);
    const Tensor& h = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& l = context->input(5);
    const Tensor& tol = context->input(6);
    const Tensor& verbose = context->input(7);
    
    
    const TensorShape& u0_shape = u0.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& tol_shape = tol.shape();
    const TensorShape& verbose_shape = verbose.shape();
    
    
    DCHECK_EQ(u0_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);
    DCHECK_EQ(verbose_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto l_tensor = l.flat<int64>().data();
    TensorShape u_shape({(*m_tensor) * (*n_tensor) * (*l_tensor)});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto u0_tensor = u0.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();

    auto tol_tensor = tol.flat<double>().data();
    auto verbose_tensor = verbose.flat<bool>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    Eikonal3D::forward(
      u_tensor, u0_tensor, f_tensor, *h_tensor, *m_tensor, *n_tensor, *l_tensor, *tol_tensor, *verbose_tensor
    );

  }
};
REGISTER_KERNEL_BUILDER(Name("EikonalThreeD").Device(DEVICE_CPU), EikonalThreeDOp);



class EikonalThreeDGradOp : public OpKernel {
private:
  
public:
  explicit EikonalThreeDGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& u0 = context->input(2);
    const Tensor& f = context->input(3);
    const Tensor& h = context->input(4);
    const Tensor& m = context->input(5);
    const Tensor& n = context->input(6);
    const Tensor& l = context->input(7);
    const Tensor& tol = context->input(8);
    const Tensor& verbose = context->input(9);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& u0_shape = u0.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& tol_shape = tol.shape();
    const TensorShape& verbose_shape = verbose.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(u0_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);
    DCHECK_EQ(verbose_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u0_shape(u0_shape);
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_h_shape(h_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_l_shape(l_shape);
    TensorShape grad_tol_shape(tol_shape);
    TensorShape grad_verbose_shape(verbose_shape);
            
    // create output tensor
    
    Tensor* grad_u0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u0_shape, &grad_u0));
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_f_shape, &grad_f));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_h_shape, &grad_h));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_l_shape, &grad_l));
    Tensor* grad_tol = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_tol_shape, &grad_tol));
    Tensor* grad_verbose = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_verbose_shape, &grad_verbose));
    
    // get the corresponding Eigen tensors for data access
    
    auto u0_tensor = u0.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto l_tensor = l.flat<int64>().data();
    auto tol_tensor = tol.flat<double>().data();
    auto verbose_tensor = verbose.flat<bool>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_u0_tensor = grad_u0->flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();
    auto grad_tol_tensor = grad_tol->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    Eikonal3D::backward(
      grad_u0_tensor, grad_f_tensor, grad_u_tensor, u_tensor, u0_tensor, f_tensor, 
      *h_tensor, *m_tensor, *n_tensor, *l_tensor
    );
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EikonalThreeDGrad").Device(DEVICE_CPU), EikonalThreeDGradOp);
