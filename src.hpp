#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // Build K (first i+1 keys) and V (first i+1 values) stacks in HBM
    Matrix *K_acc = matrix_memory_allocator.Allocate("K_acc_0");
    gpu_sim.Copy(keys[0], K_acc, Position::kInGpuHbm);
    for (size_t k = 1; k <= i; ++k) {
      Matrix *K_next = matrix_memory_allocator.Allocate("K_acc_next");
      gpu_sim.Concat(K_acc, keys[k], K_next, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(K_acc);
      K_acc = K_next;
    }
    Matrix *V_acc = matrix_memory_allocator.Allocate("V_acc_0");
    gpu_sim.Copy(values[0], V_acc, Position::kInGpuHbm);
    for (size_t k = 1; k <= i; ++k) {
      Matrix *V_next = matrix_memory_allocator.Allocate("V_acc_next");
      gpu_sim.Concat(V_acc, values[k], V_next, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(V_acc);
      V_acc = V_next;
    }

    // Move operands to SRAM and prepare K^T
    Matrix *Q = current_query;
    gpu_sim.MoveMatrixToSharedMem(Q);
    gpu_sim.MoveMatrixToSharedMem(K_acc);
    gpu_sim.MoveMatrixToSharedMem(V_acc);
    gpu_sim.Transpose(K_acc, Position::kInSharedMemory);

    // scores = Q * K^T
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(Q, K_acc, scores);

    // Softmax per row then multiply by V
    size_t m = Q->GetRowNum();
    Matrix *ans_acc = nullptr;
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(scores, row_idx, row, Position::kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.MatExp(row, exp_row);

      Matrix *denom = matrix_memory_allocator.Allocate("denom");
      gpu_sim.Sum(exp_row, denom);

      Matrix *prob_row = matrix_memory_allocator.Allocate("prob_row");
      gpu_sim.MatDiv(exp_row, denom, prob_row);

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(denom);

      Matrix *out_row = matrix_memory_allocator.Allocate("out_row");
      gpu_sim.MatMul(prob_row, V_acc, out_row);
      gpu_sim.ReleaseMatrix(prob_row);

      if (ans_acc == nullptr) {
        ans_acc = out_row;
      } else {
        Matrix *ans_next = matrix_memory_allocator.Allocate("ans_next");
        gpu_sim.Concat(ans_acc, out_row, ans_next, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(ans_acc);
        gpu_sim.ReleaseMatrix(out_row);
        ans_acc = ans_next;
      }
    }

    // Move final answer to HBM and release temporaries
    gpu_sim.MoveMatrixToGpuHbm(ans_acc);
    gpu_sim.ReleaseMatrix(K_acc);
    gpu_sim.ReleaseMatrix(V_acc);
    gpu_sim.ReleaseMatrix(scores);

    // Execute and commit answer
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ans_acc);
    gpu_sim.Run(false, &matrix_memory_allocator);
    //rater.CommitAnswer(YOUR_ANSWER_MATRIX)(Commit after running the simulator.)
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu