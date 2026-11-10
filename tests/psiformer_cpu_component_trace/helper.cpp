// Standalone host-only test; no QMCPACK global timers or scientific runtime.
#include "QMCWaveFunctions/PsiFormer/PsiFormerCPUComponentTrace.h"
#include <iostream>
#include <thread>
#include <vector>

int main()
{
  namespace trace = pf::cpu_component_trace;
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 4; ++worker)
    threads.emplace_back([] {
      for (int call = 0; call < 7; ++call)
      {
        trace::Scope scope(trace::Stage::value_envelope);
        scope.next(trace::Stage::value_head);
        scope.stop();
        scope.stop(); // Explicit stop and destruction must not double count.
      }
      try
      {
        trace::Scope scope(trace::Stage::value_determinant);
        throw 42; // An unwound stage still counts one attempted operation.
      }
      catch (int) {}
    });
  for (auto& thread : threads) thread.join();
  // Worker records must survive thread teardown. Repeated snapshots are cumulative.
  std::cout << trace::flushAllThreads() << '\n';
  std::cout << trace::flushAllThreads() << '\n';
}
