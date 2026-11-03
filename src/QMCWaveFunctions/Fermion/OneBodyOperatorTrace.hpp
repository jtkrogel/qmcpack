//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_ONE_BODY_OPERATOR_TRACE_HPP
#define QMCPLUSPLUS_ONE_BODY_OPERATOR_TRACE_HPP

#include "Configuration.h"
#include "OhmmsPETE/OhmmsMatrix.h"

#include <stdexcept>

namespace qmcplusplus
{
/** Contract a one-body operator action with a determinant inverse.
 *
 * inverse_transpose(i,j) is the determinant's stored M^{-T} element for
 * electron row i and orbital column j. action(i,j) is the corresponding
 * operator-applied orbital value at electron i. The contraction is an
 * algebraic product; complex values are not conjugated.
 */
template<typename InverseValue, typename ActionValue>
QMCTraits::FullPrecValueType oneBodyOperatorTrace(const Matrix<InverseValue>& inverse_transpose,
                                                  const Matrix<ActionValue>& action)
{
  if (inverse_transpose.rows() != action.rows() || inverse_transpose.cols() != action.cols())
    throw std::invalid_argument("one-body operator action and determinant inverse dimensions differ");

  QMCTraits::FullPrecValueType trace{};
  for (int electron = 0; electron < inverse_transpose.rows(); ++electron)
    for (int orbital = 0; orbital < inverse_transpose.cols(); ++orbital)
      trace += static_cast<QMCTraits::FullPrecValueType>(inverse_transpose(electron, orbital)) *
          static_cast<QMCTraits::FullPrecValueType>(action(electron, orbital));
  return trace;
}
} // namespace qmcplusplus

#endif
