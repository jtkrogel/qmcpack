//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//////////////////////////////////////////////////////////////////////////////////////

#include <complex>
#include <catch2/catch_test_macros.hpp>
#include "type_traits/template_types.hpp"


namespace qmcplusplus
{

TEST_CASE("makeRefVector", "[type_traits]")
{
  struct Dummy
  {
    double d;
    std::string s;
  };

  struct DerivedDummy : Dummy
  {
    float f;
  };

  std::vector<DerivedDummy> ddvec;
  for (int i = 0; i < 3; ++i)
    ddvec.push_back(DerivedDummy());

  auto bdum2 = makeRefVector<Dummy>(ddvec);
  CHECK(std::is_same<RefVector<Dummy>, decltype(bdum2)>::value);

  makeRefVector<decltype(ddvec)::value_type>(ddvec);
}

TEST_CASE("convertUPtrToRefvector", "[type_traits]")
{
  struct Dummy
  {
    double d;
    std::string s;
  };

  struct DerivedDummy : public Dummy
  {
    DerivedDummy() : e(0) {}
    double e;
  };

  struct OtherDummy
  {
    double f;
  };

  UPtrVector<Dummy> uvec;
  for (int i = 0; i < 3; ++i)
    uvec.emplace_back(std::make_unique<Dummy>());

  RefVector<Dummy> rdum(convertUPtrToRefVector(uvec));
  auto rdum2 = convertUPtrToRefVector(uvec);
  CHECK(std::is_same_v<decltype(rdum), decltype(rdum2)>);
  // Testing to make sure meta programming stops potential ambiguous template resolution.
  UPtrVector<DerivedDummy> ddv;
  ddv.emplace_back(std::make_unique<DerivedDummy>());
  ddv.emplace_back(std::make_unique<DerivedDummy>());

  auto dummy_ref_vec = convertUPtrToRefVector(ddv);
  auto d_ref_vec     = convertUPtrToRefVector<Dummy>(ddv);
  auto dd_ref_vec    = convertUPtrToRefVector(ddv);
  CHECK(!std::is_same_v<decltype(d_ref_vec), decltype(dd_ref_vec)>);

  // This should cause a compilation error. a DerivedDummy cannot be converted to an OtherDummy.
  // RefVector<OtherDummy> od_ref_vec = convertUPtrToRefVector<OtherDummy>(ddv);
}

TEST_CASE("convertPtrToRefvectorSubset", "[type_traits]")
{
  struct Dummy2
  {
    Dummy2(int j) : i(j) {}
    int i;
  };

  std::vector<Dummy2*> pvec;
  for (int i = 0; i < 5; ++i)
    pvec.push_back(new Dummy2(i));

  auto rdum = convertPtrToRefVectorSubset(pvec, 1, 4);

  CHECK(rdum.size() == 4);
  CHECK(rdum[0].get().i == 1);

  for (int i = 0; i < 5; ++i)
    delete pvec[i];
}

TEST_CASE("RefVectorWithLeader rebinds references without changing storage", "[type_traits]")
{
  struct RebindProbe
  {
    explicit RebindProbe(int initial_value) : value(initial_value) {}

    RebindProbe(const RebindProbe&) = default;

    RebindProbe& operator=(const RebindProbe& other) noexcept
    {
      value = other.value;
      ++assignment_count;
      return *this;
    }

    int value;
    int assignment_count = 0;
  };

  RebindProbe original_leader(1);
  RebindProbe replacement_leader(2);
  RebindProbe original_element(3);
  RebindProbe untouched_element(4);
  RebindProbe replacement_element(5);

  RefVectorWithLeader<RebindProbe>::BaseVec element_refs{std::ref(original_element), std::ref(untouched_element)};
  RefVectorWithLeader<RebindProbe> refs(original_leader, std::move(element_refs));

  const auto* const data_before       = refs.data();
  const std::size_t size_before       = refs.size();
  const std::size_t capacity_before   = refs.capacity();
  const int original_leader_value     = original_leader.value;
  const int original_element_value    = original_element.value;
  const int replacement_leader_value  = replacement_leader.value;
  const int replacement_element_value = replacement_element.value;

  static_assert(noexcept(refs.rebindLeader(replacement_leader)));
  static_assert(noexcept(refs.rebindElement(0, replacement_element)));

  refs.rebindLeader(replacement_leader);
  refs.rebindElement(0, replacement_element);

  CHECK(&refs.getLeader() == &replacement_leader);
  CHECK(&refs[0] == &replacement_element);
  CHECK(&refs[1] == &untouched_element);

  CHECK(refs.data() == data_before);
  CHECK(refs.size() == size_before);
  CHECK(refs.capacity() == capacity_before);

  // Rebinding must replace reference_wrapper targets rather than assign through them.
  CHECK(original_leader.value == original_leader_value);
  CHECK(original_element.value == original_element_value);
  CHECK(replacement_leader.value == replacement_leader_value);
  CHECK(replacement_element.value == replacement_element_value);
  CHECK(original_leader.assignment_count == 0);
  CHECK(original_element.assignment_count == 0);
  CHECK(replacement_leader.assignment_count == 0);
  CHECK(replacement_element.assignment_count == 0);
}
} // namespace qmcplusplus
