/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_tensor_dense_matmul_op.h"

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tindices>
class SparseTensorDenseMatMulOp : public OpKernel {
 public:
  explicit SparseTensorDenseMatMulOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_b", &adjoint_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* a_indices;
    const Tensor* a_values;
    const Tensor* a_shape;
    const Tensor* b;
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices));
    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values));
    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape));
    OP_REQUIRES_OK(ctx, ctx->input("b", &b));

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b->shape()),
                errors::InvalidArgument("Tensor 'b' is not a matrix"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_shape->shape()),
                errors::InvalidArgument("Tensor 'a_shape' is not a vector"));

    OP_REQUIRES(
        ctx, a_shape->NumElements() == 2,
        errors::InvalidArgument("Tensor 'a_shape' must have 2 elements"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_values->shape()),
                errors::InvalidArgument("Tensor 'a_values' is not a vector"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a_indices->shape()),
                errors::InvalidArgument("Tensor 'a_indices' is not a matrix"));

    const int64 nnz = a_indices->shape().dim_size(0);
    OP_REQUIRES(ctx, nnz == a_values->NumElements(),
                errors::InvalidArgument("Number of rows of a_indices does not "
                                        "match number of entries in a_values"));

    OP_REQUIRES(
        ctx, a_indices->shape().dim_size(1) == a_shape->NumElements(),
        errors::InvalidArgument("Number of columns of a_indices does not match "
                                "number of entries in a_shape"));

    auto a_shape_t = a_shape->vec<int64>();
    const int64 outer_left = (adjoint_a_) ? a_shape_t(1) : a_shape_t(0);
    const int64 outer_right =
        (adjoint_b_) ? b->shape().dim_size(0) : b->shape().dim_size(1);
    const int64 inner_left = (adjoint_a_) ? a_shape_t(0) : a_shape_t(1);
    const int64 inner_right =
        (adjoint_b_) ? b->shape().dim_size(1) : b->shape().dim_size(0);

    OP_REQUIRES(
        ctx, inner_right == inner_left,
        errors::InvalidArgument(
            "Cannot multiply A and B because inner dimension does not match: ",
            inner_left, " vs. ", inner_right,
            ".  Did you forget a transpose?  "
            "Dimensions of A: [",
            a_shape_t(0), ", ", a_shape_t(1),
            ").  Dimensions of B: ", b->shape().DebugString()));

    if (std::is_same<Device, GPUDevice>::value) {
      // The GPU implementation is optimized to use 32 bit indexing, so
      // give a friendly error to the programmer early on if they
      // exceed.
      const int int32max = std::numeric_limits<int>::max();
      OP_REQUIRES(
          ctx,
          (FastBoundsCheck(inner_left, int32max) &&
           FastBoundsCheck(inner_right, int32max) &&
           FastBoundsCheck(outer_left, int32max) &&
           FastBoundsCheck(outer_right, int32max) &&
           FastBoundsCheck(b->NumElements(), int32max) &&
           FastBoundsCheck(outer_left * outer_right, int32max) &&
           FastBoundsCheck(a_values->NumElements(), int32max)),
          errors::InvalidArgument("Cannot use GPU for > 2^31 entry inputs"));
      OP_REQUIRES(ctx, FastBoundsCheck(nnz * outer_right, int32max),
                  errors::InvalidArgument(
                      "Cannot use GPU when output.shape[1] * nnz(a) > 2^31"));
    }

    TensorShape out_shape({outer_left, outer_right});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a_values->NumElements() == 0 || b->NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

#define MAYBE_ADJOINT(ADJ_A, ADJ_B)                                        \
  if (adjoint_a_ == ADJ_A && adjoint_b_ == ADJ_B) {                        \
    Status functor_status = functor::SparseTensorDenseMatMulFunctor<       \
        Device, T, Tindices, ADJ_A,                                        \
        ADJ_B>::Compute(ctx, ctx->eigen_device<Device>(), out->matrix<T>(),     \
                        a_indices->matrix<Tindices>(), a_values->vec<T>(), \
                        b->matrix<T>());                                   \
    OP_REQUIRES_OK(ctx, functor_status);                                   \
  }

    MAYBE_ADJOINT(false, false);
    MAYBE_ADJOINT(false, true);
    MAYBE_ADJOINT(true, false);
    MAYBE_ADJOINT(true, true);

#undef MAYBE_ADJOINT
  }

 private:
  bool adjoint_a_;
  bool adjoint_b_;
};

#define REGISTER_CPU(TypeT, TypeIndex)           \
  REGISTER_KERNEL_BUILDER(                       \
      Name("SparseTensorDenseMatMul")            \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<TypeT>("T")            \
          .TypeConstraint<TypeIndex>("Tindices") \
          .HostMemory("a_shape"),                \
      SparseTensorDenseMatMulOp<CPUDevice, TypeT, TypeIndex>);

#define REGISTER_KERNELS_CPU(T) \
  REGISTER_CPU(T, int64);       \
  REGISTER_CPU(T, int32)

REGISTER_KERNELS_CPU(float);
REGISTER_KERNELS_CPU(double);
REGISTER_KERNELS_CPU(int32);
REGISTER_KERNELS_CPU(complex64);
REGISTER_KERNELS_CPU(complex128);
REGISTER_KERNELS_CPU(bfloat16);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {
#define DECLARE_GPU_SPEC(T, Tindices, ADJ_A, ADJ_B)                       \
  template <>                                                             \
  Status SparseTensorDenseMatMulFunctor<                                  \
      GPUDevice, T, Tindices, ADJ_A,                                      \
      ADJ_B>::Compute(OpKernelContext* ctx, const GPUDevice& d, typename TTypes<T>::Matrix out, \
                      TTypes<Tindices>::ConstMatrix a_indices,            \
                      typename TTypes<T>::ConstVec a_values,              \
                      typename TTypes<T>::ConstMatrix b);                 \
  extern template struct SparseTensorDenseMatMulFunctor<                  \
      GPUDevice, T, Tindices, ADJ_A, ADJ_B>;

#define REGISTER_GPU_SPEC(T, ADJ_A, ADJ_B)  \
  DECLARE_GPU_SPEC(T, int32, ADJ_A, ADJ_B); \
  DECLARE_GPU_SPEC(T, int64, ADJ_A, ADJ_B)

#define DECLARE_ADJOINT_GPU_SPEC(T)  \
  REGISTER_GPU_SPEC(T, false, false) \
  REGISTER_GPU_SPEC(T, false, true)  \
  REGISTER_GPU_SPEC(T, true, false)  \
  REGISTER_GPU_SPEC(T, true, true)

DECLARE_ADJOINT_GPU_SPEC(float);
#undef DECLARE_ADJOINT_GPU_SPEC
#undef DECLARE_GPU_SPEC
#undef REGISTER_GPU_SPEC

}  // namespace functor

#define REGISTER_GPU(TypeT, TypeIndex)           \
  REGISTER_KERNEL_BUILDER(                       \
      Name("SparseTensorDenseMatMul")            \
          .Device(DEVICE_GPU)                    \
          .TypeConstraint<TypeT>("T")            \
          .TypeConstraint<TypeIndex>("Tindices") \
          .HostMemory("a_shape"),                \
      SparseTensorDenseMatMulOp<GPUDevice, TypeT, TypeIndex>);

#define REGISTER_KERNELS_GPU(T) \
  REGISTER_GPU(T, int64);       \
  REGISTER_GPU(T, int32)

REGISTER_KERNELS_GPU(float);
#undef REGISTER_GPU
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

namespace {
Status KOutOfBoundsError(int64 k, std::size_t i, int rhs_index_a,
                         std::size_t lhs_right) {
  return errors::InvalidArgument("k (", k, ") from index[", i, ",", rhs_index_a,
                                 "] out of bounds (>=", lhs_right, ")");
}

Status MOutOfBoundsError(int64 m, std::size_t i, int lhs_index_a,
                         int64 out_dim0) {
  return errors::InvalidArgument("m (", m, ") from index[", i, ",", lhs_index_a,
                                 "] out of bounds (>=", out_dim0, ")");
}
}  // namespace

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatMulFunctor<CPUDevice, T, Tindices, ADJ_A, ADJ_B> {
  // Vectorize certain operations above this size.
  static constexpr std::size_t kNumVectorize = 32;
  // Maximum number of shards into which to divide the computation for each COO
  // Sparse Matrix instance.
  // Number of shards allocated to each thread.
  static constexpr int32 kNumShardsPerThread = 1;

  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixMap = Eigen::Map<Matrix>;
  static Status Compute(OpKernelContext* ctx, const CPUDevice& d, typename TTypes<T>::Matrix out,
                        typename TTypes<Tindices>::ConstMatrix a_indices,
                        typename TTypes<T>::ConstVec a_values,
                        typename TTypes<T>::ConstMatrix b) {
    const std::size_t nnz = a_values.size();
    const std::size_t rhs_right = (ADJ_B ? b.dimension(0) : b.dimension(1));
    const std::size_t lhs_right = (ADJ_B ? b.dimension(1) : b.dimension(0));
    const int lhs_index_a = ADJ_A ? 1 : 0;
    const int rhs_index_a = ADJ_A ? 0 : 1;
    static constexpr int32 kMinShards = 10;

    out.setZero();

    // TODO(ebrevdo): After many failed experiments, can't find a multi-threaded
    // approach that achieves the performance of the single threaded
    // one.  Perhaps Eigen threadpool implementation is just too slow?
    //
    //
    if (nnz > 4096) {
//     if (false) {
      const int b_chip_index = ADJ_B ? 1 : 0;
      auto maybe_adjoint_b = MaybeAdjoint<decltype(b), ADJ_B>(b);
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      const int32 num_threads = worker_threads.num_threads;
      // Each thread writes to its own copy of the matrix product. These
      // `num_threads` copies are summed together to obtain the final result.
      Tensor matmul_result_buffer;
//       OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
//                                              TensorShape({num_threads + 1,
//                                                           out->dimension(0),
//                                                           out->dimension(1)}),
//                                              &matmul_result_buffer));
       ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape({(num_threads + 1)*out.dimension(0),
                                                          out.dimension(1)}),
                                             &matmul_result_buffer);
       int outNumElements = out.dimension(0) * out.dimension(1);
       int outDimension1 = out.dimension(1);
       functor::SetZeroFunctor<CPUDevice, T> set_zero;
       set_zero(d, matmul_result_buffer.flat<T>());

       LOG(INFO) << "lhs_index_a=" << lhs_index_a << ", rhs_index_a=" << rhs_index_a << ", lhs_right=" << lhs_right << ", rhs_right="<< rhs_right;
       const int64 block_size = std::max(int64(64), int64(nnz /(kNumShardsPerThread * num_threads)));
//       const int64 block_size = 16384;
//           nnz / num_threads;
      auto lambda = [&](Tindices block_begin, Tindices block_end, int tid) {
        LOG(INFO) << "block_begin=" << block_begin << ", block_end=" << block_end << ", tid=" << tid;
        // [<lhs_index_a>m, <rhs_index_a>k] @ [<lhs_right>k, <rhs_right>n] = [m, n]
//         auto output_tensor = matmul_result_buffer.tensor<T, 3>().template chip<0>(tid);

        
        for (long long int i = block_begin; i < block_end; ++i) {
                const Tindices m = internal::SubtleMustCopy(a_indices(i, lhs_index_a));
                const Tindices k = internal::SubtleMustCopy(a_indices(i, rhs_index_a));
                if (!FastBoundsCheck(k, lhs_right)) {
                    LOG(INFO) << KOutOfBoundsError(k, i, rhs_index_a, lhs_right);
                    continue;
                }
                if (!FastBoundsCheck(m, out.dimension(0))) {
                  LOG(INFO) << MOutOfBoundsError(m, i, lhs_index_a, out.dimension(0));
                  continue;
                }

                const T a_value = ADJ_A ? MaybeConj(a_values(i)) : a_values(i);
//                 MatrixMap output_map(
//                     matmul_result_buffer.flat<T>().data() +
//                     tid * block_size * outNumElements +
//                     m * out.dimension(1),
//                     out.dimension(1), 1);

                matmul_result_buffer.matrix<T>().template chip<0>(tid*out.dimension(0)+m) += b.template chip<b_chip_index>(k) * a_value;
//                 output_map.noalias() += (b.template chip<b_chip_index>(k) * a_value).matrix<T>();
//                 out.template chip<0>(m) += b.template chip<b_chip_index>(k) * a_value;

//                 for (std::size_t n = 0; n < rhs_right; ++n) {
//                   const T b_value = maybe_adjoint_b(k, n);
// // //                   fixme
// //                   matmul_result_buffer.flat<T>().data()[tid * block_size * outNumElements + m * outDimension1 + n] += a_value * b_value;
//                 }
              }
        return;
            };

      worker_threads.workers->ParallelForWithWorkerId(
          nnz  /* total */,
          thread::ThreadPool::SchedulingParams(
              thread::ThreadPool::SchedulingStrategy::
                  kFixedBlockSize /* strategy */,
              absl::nullopt /* cost_per_unit */, block_size),
          lambda
          );

      // Sum across each thread's matmul result.
      using Reducer = Eigen::internal::SumReducer<T>;
      using Index = typename TTypes<T>::Tensor::Index;
//       // dim not match???
//       out = matmul_result_buffer.matrix<T>().reduce(
//           Eigen::array<Index, 1>({0}), Reducer());

    } else if (rhs_right < kNumVectorize) {
      // Disable vectorization if the RHS of output is too small
      auto maybe_adjoint_b = MaybeAdjoint<decltype(b), ADJ_B>(b);

      for (std::size_t i = 0; i < nnz; ++i) {
        const Tindices m = internal::SubtleMustCopy(a_indices(i, lhs_index_a));
        const Tindices k = internal::SubtleMustCopy(a_indices(i, rhs_index_a));
        if (!FastBoundsCheck(k, lhs_right)) {
          return KOutOfBoundsError(k, i, rhs_index_a, lhs_right);
        }
        if (!FastBoundsCheck(m, out.dimension(0))) {
          return MOutOfBoundsError(m, i, lhs_index_a, out.dimension(0));
        }
        const T a_value = ADJ_A ? MaybeConj(a_values(i)) : a_values(i);
        for (std::size_t n = 0; n < rhs_right; ++n) {
          const T b_value = maybe_adjoint_b(k, n);
          out(m, n) += a_value * b_value;
        }
      }
    } else {
      // Vectorization via Eigen.
      const int b_chip_index = ADJ_B ? 1 : 0;

#define LOOP_NNZ(b_passed)                                                  \
  for (std::size_t i = 0; i < nnz; ++i) {                                   \
    const Tindices m = internal::SubtleMustCopy(a_indices(i, lhs_index_a)); \
    const Tindices k = internal::SubtleMustCopy(a_indices(i, rhs_index_a)); \
    const T a_value = (ADJ_A) ? MaybeConj(a_values(i)) : a_values(i);       \
    if (!FastBoundsCheck(k, lhs_right)) {                                   \
      return KOutOfBoundsError(k, i, rhs_index_a, lhs_right);               \
    }                                                                       \
    if (!FastBoundsCheck(m, out.dimension(0))) {                            \
      return MOutOfBoundsError(m, i, lhs_index_a, out.dimension(0));        \
    }                                                                       \
    out.template chip<0>(m) +=                                              \
        b_passed.template chip<b_chip_index>(k) * a_value;                  \
  }

      if (ADJ_B) {
        // Perform transpose and conjugation on B once, since we chip out B's
        // columns in the nnz loop.
        Eigen::array<int, 2> shuffle(1, 0);  // preserve dimension order
        Eigen::Tensor<T, 2, Eigen::ColMajor> col_major_conj_b =
            b.swap_layout().shuffle(shuffle).conjugate();
        LOOP_NNZ(col_major_conj_b);
      } else {
        LOOP_NNZ(b);
      }
#undef LOOP_NNZ
    }
    return Status::OK();
  }
};

}  // namespace functor

}  // namespace tensorflow
