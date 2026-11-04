//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include "MFRMetaFields.h"
#include "OhmmsData/AttributeSet.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminantBase.h"
#include "Utilities/TimerManager.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qmcplusplus
{
namespace
{
// Release builds may enable fast-math; inspect incoming IEEE exponent bits.
bool finiteField(double value)
{
  static_assert(sizeof(double)==sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
  std::uint64_t bits;
  std::memcpy(&bits,&value,sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

struct Handle
{
  hid_t id;
  herr_t (*close)(hid_t);
  Handle(hid_t value, herr_t (*closer)(hid_t)) : id(value), close(closer)
  {
    if (id<0) throw std::runtime_error("MFRMetaFields missing or unreadable HDF5 object");
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() { close(id); }
};

std::string text(hid_t object, const char* name)
{
  Handle attribute(H5Aopen(object,name,H5P_DEFAULT),H5Aclose);
  Handle space(H5Aget_space(attribute.id),H5Sclose);
  Handle type(H5Aget_type(attribute.id),H5Tclose);
  const size_t size=H5Tget_size(type.id);
  if (H5Sget_simple_extent_ndims(space.id)!=0 || H5Tget_class(type.id)!=H5T_STRING ||
      H5Tis_variable_str(type.id)!=0 || size==0 || size>256)
    throw std::runtime_error("MFRMetaFields requires bounded fixed-string scalar attributes");
  std::vector<char> bytes(size+1,0);
  if (H5Aread(attribute.id,type.id,bytes.data())<0)
    throw std::runtime_error("MFRMetaFields cannot read string attribute");
  std::string result(bytes.data(),size);
  result.resize(result.find('\0')==std::string::npos ? result.size() : result.find('\0'));
  while (!result.empty() && result.back()==' ') result.pop_back();
  return result;
}

std::vector<std::shared_ptr<UBspline_3d_d>> readField(hid_t file, const char* name,
                                                    const char* units, int spins)
{
  Handle group(H5Gopen2(file,(std::string("/fields/")+name).c_str(),H5P_DEFAULT),H5Gclose);
  H5G_info_t info;
  if (H5Gget_info(group.id,&info)<0 || info.nlinks!=spins)
    throw std::runtime_error("MFRMetaFields field has inconsistent spin channels");
  if (text(group.id,"units")!=units || text(group.id,"grid_order")!="HDF5: z,y,x; x fastest" ||
      text(group.id,"grid_origin")!="fractional coordinates (0,0,0)" ||
      text(group.id,"periodic_grid_endpoint")!="excluded")
    throw std::runtime_error("MFRMetaFields unsupported field units/grid convention");
  Handle attribute(H5Aopen(group.id,"grid_dimensions",H5P_DEFAULT),H5Aclose);
  Handle space(H5Aget_space(attribute.id),H5Sclose);
  Handle type(H5Aget_type(attribute.id),H5Tclose);
  hsize_t length=0;
  if (H5Sget_simple_extent_ndims(space.id)!=0 || H5Tget_class(type.id)!=H5T_ARRAY ||
      H5Tget_array_ndims(type.id)!=1 || H5Tget_array_dims2(type.id,&length)<0 || length!=3)
    throw std::runtime_error("MFRMetaFields requires scalar length-three grid-dimension array");
  Handle base(H5Tget_super(type.id),H5Tclose);
  if (H5Tget_class(base.id)!=H5T_INTEGER || H5Tget_size(base.id)!=4)
    throw std::runtime_error("MFRMetaFields requires int32 grid dimensions");
  Handle memory(H5Tarray_create2(H5T_NATIVE_INT,1,&length),H5Tclose);
  std::array<int,3> dimensions;
  if (H5Aread(attribute.id,memory.id,dimensions.data())<0)
    throw std::runtime_error("MFRMetaFields cannot read grid dimensions");
  size_t count=1;
  for (int n:dimensions)
  {
    if (n<4 || count>std::numeric_limits<size_t>::max()/sizeof(double)/static_cast<size_t>(n))
      throw std::runtime_error("MFRMetaFields invalid or overflowing spline dimensions");
    count*=n;
  }
  std::vector<std::shared_ptr<UBspline_3d_d>> splines;
  for (int spin=0;spin<spins;++spin)
  {
    Handle dataset(H5Dopen2(group.id,("spin_"+std::to_string(spin+1)).c_str(),H5P_DEFAULT),H5Dclose);
    Handle data_type(H5Dget_type(dataset.id),H5Tclose);
    Handle data_space(H5Dget_space(dataset.id),H5Sclose);
    std::array<hsize_t,3> extent;
    if (H5Tget_class(data_type.id)!=H5T_FLOAT || H5Tget_size(data_type.id)!=8 ||
        H5Sget_simple_extent_ndims(data_space.id)!=3 ||
        H5Sget_simple_extent_dims(data_space.id,extent.data(),nullptr)<0 ||
        extent!=std::array<hsize_t,3>{static_cast<hsize_t>(dimensions[2]),
                                     static_cast<hsize_t>(dimensions[1]),static_cast<hsize_t>(dimensions[0])})
      throw std::runtime_error("MFRMetaFields requires binary64 (nz,ny,nx) field data");
    std::vector<double> values(count),ordered(count);
    if (H5Dread(dataset.id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,values.data())<0)
      throw std::runtime_error("MFRMetaFields cannot read field data");
    for (int i=0;i<dimensions[0];++i)
      for (int j=0;j<dimensions[1];++j)
        for (int k=0;k<dimensions[2];++k)
        {
          const double value=values[(static_cast<size_t>(k)*dimensions[1]+j)*dimensions[0]+i];
          if (!finiteField(value)) throw std::runtime_error("MFRMetaFields non-finite field value");
          ordered[(static_cast<size_t>(i)*dimensions[1]+j)*dimensions[2]+k]=value;
        }
    Ugrid grids[3];
    for (int axis=0;axis<3;++axis) grids[axis]={0.0,1.0,dimensions[axis]};
    BCtype_d bc{PERIODIC,PERIODIC,0.0,0.0};
    std::shared_ptr<UBspline_3d_d> spline(create_UBspline_3d_d(grids[0],grids[1],grids[2],bc,bc,bc,ordered.data()),
                                         destroy_Bspline);
    if (!spline) throw std::runtime_error("MFRMetaFields spline allocation failed");
    splines.push_back(std::move(spline));
  }
  return splines;
}
constexpr const char* observable_names[]={"MFRMetaU","MFRMetaDifferential","MFRMetaDifferentialImag",
                                         "TotalEnergyMF","IonIonMF"};
} // namespace

MFRMetaFields::MFRMetaFields(ParticleSet& pset, TrialWaveFunction* psi)
    : reference_(std::make_unique<MFRCoreAction>(pset,psi,MFRCoreAction::ReferencePayload::META_FIELDS))
{
  setEnergyDomain(POTENTIAL);
  oneBodyQuantumDomain(pset);
}

bool MFRMetaFields::put(xmlNodePtr cur)
{
  reference_->put(cur);
  OhmmsAttributeSet attributes;
  attributes.add(filename_,"href");
  attributes.put(cur);
  Handle file(H5Fopen(filename_.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
  auto fields=std::make_shared<Fields>();
  const int spins=reference_->getReferenceSpinChannels();
  fields->u=readField(file.id,"u","hartree",spins);
  fields->q=readField(file.id,"q","dimensionless",spins);
  fields_=std::move(fields);
  value_=differential_=imaginary_differential_=0.0;
  return true;
}

MFRMetaFields::FieldValues MFRMetaFields::evaluateFields(int spin,const PosType& position) const
{
  if (!fields_ || spin<0 || spin>=fields_->u.size())
    throw std::runtime_error("MFRMetaFields requested an unavailable spin channel");
  const auto& lattice=reference_->getReferenceLattice();
  const auto r=lattice.toUnit_floor(QMCTraits::QTFull::PosType(position));
  FieldValues result;
  double fractional_gradient[3];
  eval_UBspline_3d_d(fields_->u[spin].get(),r[0],r[1],r[2],&result.u);
  eval_UBspline_3d_d_vg(fields_->q[spin].get(),r[0],r[1],r[2],&result.q,fractional_gradient);
  result.grad_q=0.0;
  for (int cart=0;cart<3;++cart)
    for (int reduced=0;reduced<3;++reduced)
      result.grad_q[cart]+=lattice.getG()(cart,reduced)*fractional_gradient[reduced];
  return result;
}

MFRMetaFields::FullPrecValueType MFRMetaFields::evaluateTrace(TrialWaveFunction& psi,ParticleSet& pset)
{
  reference_->ensureReference(psi);
  psi.completeUpdates();
  const auto determinants=psi.findSD();
  if (!psi.findMSD().empty() || determinants.size()!=1)
    throw std::runtime_error("MFRMetaFields requires a single Slater determinant product");
  static thread_local auto* trace_timer = &createGlobalTimer("MFR::MetaFieldsTrace", timer_level_fine);
  ScopedTimer scope(*trace_timer);
  // As for BareKineticEnergy, accepted P.G/P.L must be current at this
  // energy boundary. Subtract only cached nonfermionic log derivatives;
  // rebuilding the determinant or its orbital VGL is unnecessary.
  nonfermion_g_.resize(pset.getTotalNum());
  nonfermion_l_.resize(pset.getTotalNum());
  nonfermion_g_=0.0;
  nonfermion_l_=0.0;
  for (const auto& component : psi.getOrbitals())
    if (!component->isFermionic())
      component->evaluateGL(pset,nonfermion_g_,nonfermion_l_,false);

  value_=0.0;
  FullPrecValueType total{};
  for (int electron=0;electron<pset.getTotalNum();++electron)
  {
    const int spin=fields_->u.size()==1 ? 0 : pset.GroupID[electron];
    const auto field=evaluateFields(spin,pset.R[electron]);
    value_+=field.u;
    const auto determinant_g=pset.G[electron]-nonfermion_g_[electron];
    FullPrecValueType determinant_laplacian=pset.L[electron]-nonfermion_l_[electron];
    for (int axis=0;axis<3;++axis)
      determinant_laplacian+=determinant_g[axis]*determinant_g[axis];
    FullPrecValueType action=field.q*determinant_laplacian;
    for (int axis=0;axis<3;++axis)
      action+=field.grad_q[axis]*determinant_g[axis];
    // Algebraic products, without complex conjugation; Q acts on D only.
    total-=0.5*action;
  }
  differential_=std::real(total);
  imaginary_differential_=std::imag(total);
  return total;
}

bool MFRMetaFields::get(std::ostream& os) const
{
  os << "Meta-GGA multiplicative/differential fields from " << filename_;
  return true;
}

void MFRMetaFields::addObservables(PropertySetType& plist,BufferType& collectables)
{
  for (size_t i=0;i<indices_.size();++i) indices_[i]=plist.add(observable_names[i]);
}

void MFRMetaFields::registerObservables(std::vector<ObservableHelper>& h5desc,hdf_archive& file) const
{
  for (size_t i=0;i<indices_.size();++i)
  {
    // Shared reference constants are registered once, regardless of XML order.
    if (std::any_of(h5desc.begin(),h5desc.end(),[&](const auto& h){return h.lower_bound==indices_[i];}))
      continue;
    h5desc.emplace_back(hdf_path{observable_names[i]});
    h5desc.back().set_dimensions(std::vector<int>(1,1),indices_[i]);
  }
}

void MFRMetaFields::writeProperties(PropertySetType& plist,int offset) const
{
  const FullPrecRealType values[]={value_,differential_,imaginary_differential_,getTotalEnergyMF(),getIonIonEnergyMF()};
  for (size_t i=0;i<indices_.size();++i) plist[indices_[i]+offset]=values[i];
}
void MFRMetaFields::setObservables(PropertySetType& plist) { writeProperties(plist,0); }
void MFRMetaFields::setParticlePropertyList(PropertySetType& plist,int offset) { writeProperties(plist,offset); }

std::unique_ptr<OperatorBase> MFRMetaFields::makeClone(ParticleSet& pset) const
{
  auto clone=std::make_unique<MFRMetaFields>(pset);
  auto reference=reference_->makeClone(pset);
  clone->reference_.reset(static_cast<MFRCoreAction*>(reference.release()));
  clone->fields_=fields_;
  clone->filename_=filename_;
  clone->value_=value_;
  clone->differential_=differential_;
  clone->imaginary_differential_=imaginary_differential_;
  clone->indices_=indices_;
  clone->setName(getName());
  return clone;
}
} // namespace qmcplusplus
