

# Python standard library imports
import os
import inspect
from time import process_time
from copy import deepcopy

# Non-standard Python imports
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Nexus imports
import memory
from unit_converter import convert
from generic import obj
from developer import DevBase,log,error,ci
from numerics import simstats
from grid_functions import grid_function,read_grid,grid as generate_grid
from grid_functions import SpheroidGrid
from grid_functions import ParallelotopeGrid,ParallelotopeGridFunction
from structure import Structure,read_structure
from fileio import XsfFile




class VLog(DevBase):
    """
    Functor class to handle logging with verbosity, memory, and time options.
    """

    verbosity_levels = obj(
        none = 0,
        low  = 1,
        high = 2,
        )

    def __init__(self):
        self.tstart    = process_time()
        self.tlast     = self.tstart
        self.mstart    = memory.resident(children=True)
        self.mlast     = self.mstart
        self.verbosity = self.verbosity_levels.low
        self.indent    = 0
    #end def __init__


    def __call__(self,msg,level='low',n=0,time=False,mem=False,width=75):
        if self.verbosity==self.verbosity_levels.none:
            return
        elif self.verbosity >= self.verbosity_levels[level]:
            if mem or time:
                npad = max(0,width-2*(n+self.indent)-len(msg)-36)
                if npad>0:
                    msg += npad*' '
                #end if
                if mem:
                    dm = 1e6 # MB
                    mnow = memory.resident(children=True)
                    msg += '  (mem add {:6.2f}, tot {:6.2f})'.format((mnow-self.mlast)/dm,(mnow-self.mstart)/dm)
                    self.mlast = mnow
                #end if
                if time:
                    tnow = process_time()
                    msg += '  (t elap {:7.3f}, tot {:7.3f})'.format(tnow-self.tlast,tnow-self.tstart)
                    self.tlast = tnow
                #end if
            #end if
            log(msg,n=n+self.indent)
        #end if
    #end def __init__

    def increment(self,n=1):
        self.indent += n
    #end def increment

    def decrement(self,n=1):
        self.indent -= n
    #end def decrement

    def set_none(self):
        self.verbosity = self.verbosity_levels.none
    #end def set_none

    def set_low(self):
        self.verbosity = self.verbosity_levels.low
    #end def set_low

    def set_high(self):
        self.verbosity = self.verbosity_levels.high
    #end def set_high

    def set_verbosity(level):
        if level not in self.verbosity_levels:
            vlinv = self.verbosity_levels.inverse()
            error('Cannot set verbosity level to "{}".\nValid options are: {}'.format(level,[vlinv[i] for i in sorted(vlinv.keys())]))
        #end if
        self.verbosity = self.verbosity_levels[level]
    #end def set_verbosity
#end class VLog
vlog = VLog()



def set_verbosity(level):
    """
    Set verbosity level.
    """
    vlog.set_verbosity(level)
#end def set_verbosity



class Missing:
    """
    Represents missing keyword argments to functions.
    """
    def __call__(self,value):
        return isinstance(value,Missing)
    #end def __call__
#end class Missing
missing = Missing()



class AttributeProperties(DevBase):
    """
    Descriptive properties of attributes assignable to classes derived from 
    `DefinedAttributeBase`.

    Attributes
    ----------
    assigned : `set`
        Set of names of attribute properties that have been assigned.
    name : `str`
        Name of the attribute.  This is set internally for consistency and is not
        required at the time of definition.
    dest : `str`
        Assignment destination for the value of the named attribute.  This refers
        to the name of a collection object (usually of type `obj`) at the top level
        of the class.  In this way, attributed properties of two level container 
        classes can be defined in a single level manner.
    type : `class object, optional`
        Expected class/type of the values assigned to this attribute.  If provided, 
        explicit type checks will be made upon assignment.  This allows for strong
        typing applied to any or all of an object's attributes.
    default : `optional`
        Default value to be assigned if none is provided for the attribute.  This 
        can be a simple value (e.g. an integer/float/bool), or it can be a class 
        or function.  If a class is provided as the default, it's empty constructor 
        is called at the time of default assignment.  If a function is provided as 
        the default, the value it returns will be used at the time of default 
        assignment.
    no_default : `bool, optional, default False`
        Do not assign any default value for this attribute.  If True, then the 
        default behavior is for the attribute key to be missing from the 
        instantiated object's attribute dictionary.
    deepcopy : `bool, optional, default False`
        Use a deep copy operation upon assignment for this attribute.
    required : `bool, optional, default False`
        Declare that this attribute is required.  This enables presence checks 
        upon request.
    """
    def __init__(self,**kwargs):
        self.assigned   = set(kwargs.keys())
        self.name       = kwargs.pop('name'      , None )
        self.dest       = kwargs.pop('dest'      , None )
        self.type       = kwargs.pop('type'      , None )
        self.default    = kwargs.pop('default'   , None )
        self.no_default = kwargs.pop('no_default', False)
        self.deepcopy   = kwargs.pop('deepcopy'  , False)
        self.required   = kwargs.pop('required'  , False)
        if len(kwargs)>0:
            self.error('Invalid init variable attributes received.\nInvalid attributes:\n{}\nThis is a developer error.'.format(obj(kwargs)))
        #end if
    #end def __init__
#end class AttributeProperties



class DefinedAttributeBase(DevBase):
    """
    Enables detailed attribute requirements to be enforced for derived classes.

    This includes specifications of required/optional attributes, controlled 
    attribute assignment including name/type checks and/or deep copying, use of 
    static or functional assignment of default values, and controlled retrieval 
    of attributes including presence checks.  The attribute definitions comprise 
    a namespace of allowed attributes that are strictly enforced.  In these 
    ways, the typical fully dynamic nature of objects is more strictly controlled 
    to give greater confidence in runtime correctness.

    Includes mechanisms for simple composition of attribute requirements for 
    multi-level inheritance of classes derived from this base class.
    """

    @classmethod
    def set_unassigned_default(cls,default):
        """
        Set a global default value for attributes when attribute level defaults
        are not specified.  A reasonable choice is the `None` value.
        """
        cls.unassigned_default = default
    #end def set_unassigned_default


    @classmethod
    def define_attributes(cls,*other_cls,**attribute_properties):
        """
        Main class-level interface to specify properties of each attribute.

        Listed value arguments (`other_cls`), correspond to the parent classes 
        of the current class in the inheritance hierarchy that also descend from
        `DefinedAttributeBase`.  The attribute properties of the parent classes 
        are concatenated for inclusion as requirements for the child class. 
        Currently inheritance via a single parent class is supported.

        Keyword inputs (`attribute_properties`) are key-value pairs, with the 
        key corresponding to an allowed attribute name, and the value being 
        simple dictionary inputs corresponding to the inputs of the 
        `AttributeProperties` class.

        This function assigns the class-level attributes `requried_attributes`, 
        `deepcopy_attributes`, `typed_attributes`, `toplevel_attributes`, and 
        `sublevel_attributes` which contain attribute names allowed in 
        instances of the class.
        """
        if len(other_cls)==1:
            if issubclass(other_cls[0],DefinedAttributeBase):
                cls.obtain_attributes(other_cls[0])
            else:
                cls.class_error('Parent class must derive from DefinedAttributeBase.\nThis is a developer error.')
            #end if
        elif len(other_cls)>1:
            cls.class_error('Only a single parent class is supported by DefinedAttributeBase.\nThis is a developer error.')
        #end if
        if cls.class_has('attribute_definitions'):
            attr_defs = cls.attribute_definitions
        else:
            attr_defs = obj()
            cls.class_set(
                attribute_definitions = attr_defs
                )
        #end if
        for name,attr_props in attribute_properties.items():
            attr_props = AttributeProperties(**attr_props)
            attr_props.name = name
            if name not in attr_defs:
                attr_defs[name] = attr_props
            else:
                p = attr_defs[name]
                for n in attr_props.assigned:
                    p[n] = attr_props[n]
                #end for
            #end if
        #end for
        if cls.class_has('unassigned_default'):
            for p in attr_defs:
                if 'default' not in p.assigned:
                    p.default = cls.unassigned_default
                #end if
            #end for
        #end if
        required_attributes = set()
        deepcopy_attributes = set()
        typed_attributes    = set()
        toplevel_attributes = set()
        sublevel_attributes = set()
        for name,props in attr_defs.items():
            if props.required:
                required_attributes.add(name)
            #end if
            if props.deepcopy:
                deepcopy_attributes.add(name)
            #end if
            if props.type is not None:
                typed_attributes.add(name)
            #end if
            if props.dest is None:
                toplevel_attributes.add(name)
            else:
                sublevel_attributes.add(name)
            #end if
        #end for
        cls.class_set(
            required_attributes = required_attributes,
            deepcopy_attributes = deepcopy_attributes,
            typed_attributes    = typed_attributes,
            toplevel_attributes = toplevel_attributes,
            sublevel_attributes = sublevel_attributes,
            )
    #end def define_attributes


    @classmethod
    def obtain_attributes(cls,super_cls):
        """
        Collect attribute properties from the parent class.
        """
        cls.class_set(
            attribute_definitions = super_cls.attribute_definitions.copy()
            )
    #end def obtain_attributes


    def __init__(self,**values):
        """
        Provides a general default constructor.

        Arbitrary named attributes are accepted, with defaults first set.
        Comprehensive attribute checks are left to the implementer, as 
        deemed appropriate, but are generically available via the 
        `check_attributes` member function.
        """
        if len(values)>0:
            self.set_default_attributes()
            self.set_attributes(**values)
        #end if
    #end def __init__


    def initialize(self,**values):
        """
        Enable deferred initialization following empty contruction.
        """
        self.set_default_attributes()
        if len(values)>0:
            self.set_attributes(**values)
        #end if
    #end def initialize


    def set_default_attributes(self):
        """
        Assign values to named attributes that have specified defaults.
        """
        cls = self.__class__
        props = cls.attribute_definitions
        for name in cls.toplevel_attributes:
            self._set_default_attribute(name,props[name])
        #end for
        for name in cls.sublevel_attributes:
            self._set_default_attribute(name,props[name])
        #end for
    #end def set_default_attributes


    def set_attributes(self,**values):
        """
        Assign values to attributes.

        All required attributes must be provided in a single call.  Requirements
        on each attribute, such as type, are strictly enforced.
        """
        cls = self.__class__
        value_names = set(values.keys())
        attr_names  = set(cls.attribute_definitions.keys())
        invalid     = value_names - attr_names
        if len(invalid)>0:
            v = obj()
            v.transfer_from(values,invalid)
            self.error('Attempted to set unrecognized attributes\nUnrecognized attributes:\n{}'.format(v))
        #end if
        missing = set(cls.required_attributes) - value_names
        if len(missing)>0:
            msg = ''
            for n in sorted(missing):
                msg += '\n  '+n
            #end for
            self.error('Required attributes are missing.\nPlease provide the following attributes during initialization:{}'.format(msg))
        #end if
        props = cls.attribute_definitions
        toplevel_names = value_names & cls.toplevel_attributes
        for name in toplevel_names:
            self._set_attribute(self,name,values[name],props[name])
        #end for
        sublevel_names = value_names - toplevel_names
        for name in sublevel_names:
            p = props[name]
            if p.dest not in self:
                self.error('Attribute destination "{}" does not exist at the top level.\nThis is a developer error.'.format(p.dest))
            #end if
            self._set_attribute(self[p.dest],name,values[name],p)
        #end for
    #end def set_attributes


    def check_attributes(self,exit=False):
        """
        Check the integrity of the attributes of the instance according to the 
        defined attribute properties.
        """
        msg = ''
        cls = self.__class__
        a = obj()
        for name in cls.toplevel_attributes:
            if name in self:
                a[name] = self[name]
            #end if
        #end for
        props = cls.attribute_definitions
        for name in cls.sublevel_attributes:
            p = props[name]
            if p.dest in self:
                sub = self[p.dest]
                if name in sub:
                    a[name] = sub[name]
                #end if
            #end if
        #end for
        present = set(a.keys())
        missing = cls.required_attributes - present
        if len(missing)>0:
            m = ''
            for n in sorted(missing):
                m += '\n  '+n
            #end for
            msg += 'Required attributes are missing.\nPlease provide the following attributes during initialization:{}\n'.format(m)
        #end if
        for name in cls.typed_attributes:
            if name in a:
                p = props[name]
                v = a[name]
                if not isinstance(v,p.type):
                    msg += 'Attribute "{}" has invalid type.\n  Type expected: {}\n  Type present: {}\n'.format(name,p.type.__name__,v.__class__.__name__)
                #end if
            #end if
        #end for
        valid = len(msg)==0
        if not valid and exit:
            self.error(msg)
        #end if
        return valid
    #end def check_attributes


    def check_unassigned(self,value):
        """
        Determine whether an attribute has been assigned a value.
        """
        cls = self.__class__
        unassigned = cls.class_has('unassigned_default') and value is cls.unassigned_default
        return unassigned
    #end def check_unassigned


    def set_attribute(self,name,value):
        """
        Assign a value to a named attribute.

        Assignment includes checks against the allowed namespace and, 
        if requested, type checks, deepcopying, and assignment to a nested 
        destination.
        """
        cls = self.__class__
        props = cls.attribute_definitions
        if name not in props:
            self.error('Cannot set unrecognized attribute "{}".\nValid options are: {}'.format(name,sorted(props.keys())))
        #end if
        p = props[name]
        if p.type is not None and not isinstance(value,p.type):
            self.error('Cannot set attribute "{}".\nExpected value with type: {}\nReceived value with type: {}'.format(name,p.type.__name__,value.__class__.__name__))
        #end if
        if p.deepcopy:
            value = deepcopy(value)
        #end if
        if p.dest is None:
            self[name] = value
        elif p.dest not in self:
            self.error('Cannot set attribute "{}".\nAttribute destination "{}" does not exist.'.format(name,p.dest))
        else:
            self[p.dest][name] = value
        #end if
    #end def set_attribute


    def get_attribute(self,name,value=missing,assigned=True):
        """
        Retrieve the value of a named attribute.

        Parameters
        ----------
        name : `str`
            Name of the attribute.
        value : `optional`
            Default value to be returned if the attribute has not been assigned.
        assigned : `bool, default True`
            Require that the attribute has been assigned and explicitly check 
            for assignment before returning the value.  This requirement is not 
            enforced if `value` is provided.
        """
        default_value    = value
        default_provided = not missing(default_value)
        require_assigned = assigned and not default_provided
        cls = self.__class__
        props = cls.attribute_definitions
        if name not in props:
            self.error('Cannot get unrecognized attribute "{}".\nValid options are: {}'.format(name,sorted(props.keys())))
        #end if
        p = props[name]
        value = missing
        if p.dest is None:
            if name in self:
                value = self[name]
            #end if
        elif p.dest in self and name in self[p.dest]:
            value = self[p.dest][name]
        #end if
        present = not missing(value)
        if not present and default_provided:
            return default_value
        else:
            unassigned = True
            if present:
                unassigned = self.check_unassigned(value)
            #end if
            if not present or (unassigned and require_assigned):
                extra = ''
                if p.dest is not None:
                    extra = ' at location "{}"'.format(p.dest)
                #end if
                if not present:
                    msg = 'Cannot get attribute "{}"{}.\nAttribute does not exist.'.format(name,extra)
                else:
                    msg = 'Cannot get attribute "{}"{}.\nAttribute has not been assigned.'.format(name,extra)
                #end if
                self.error(msg)
            #end if
        #end if
        return value
    #end def get_attribute


    def has_attribute(self,name):
        """
        Check for the presence of a named attribute.
        """
        return not (name not in self or self.check_unassigned(self[name]))
    #end def has_attribute


    def _set_default_attribute(self,name,props):
        """
        Internal function used to assign default values without protective interfaces.
        """
        p = props
        if p.no_default:
            return
        #end if
        value = p.default
        if inspect.isclass(value) or inspect.isfunction(value):
            value = value()
        #end if
        if p.dest is None:
            self[name] = value
        elif p.dest not in self:
            self.error('Attribute destination "{}" does not exist at the top level.\nThis is a developer error.'.format(p.dest))
        else:
            self[p.dest][name] = value
        #end if
    #end def _set_default_attribute


    def _set_attribute(self,container,name,value,props):
        """
        Internal function used to assign attribute values without protective interfaces.
        """
        p = props
        if p.type is not None and not isinstance(value,p.type):
            self.error('Cannot set attribute "{}".\nExpected value with type: {}\nReceived value with type: {}'.format(name,p.type.__name__,value.__class__.__name__))
        #end if
        if p.deepcopy:
            value = deepcopy(value)
        #end if
        container[name] = value
    #end def _set_attribute

#end class DefinedAttributeBase



class Observable(DefinedAttributeBase):
    """
    Base class for generic observables.

    Attributes
    ----------
    info : `obj`
        Container for extra information, including data for derived classes.
    initialized : `bool, default False`
        Record whether instance initialization is complete.  Contained in `info`.
    structure : `Structure, optional, default None`
        Atomic structure of the system under observation.
    """
    def __init__(self,**values):
        self.initialize(**values)
    #end def __init__

    def initialize(self,**values):
        DefinedAttributeBase.initialize(self,**values)
        if len(values)>0:
            self.set_attribute('initialized',True)
        #end if
    #end def initialize
#end class Observable

Observable.set_unassigned_default(None)

Observable.define_attributes(
    info = obj( 
        type    = obj, 
        default = obj,
        ),
    initialized = obj(
        dest    = 'info',
        type    = bool,
        default = False,
        ),
    structure = obj(
        dest     = 'info',
        type     = Structure,
        default  = None,
        deepcopy = True,
        ),
    )



class ObservableWithComponents(Observable):
    """
    Base class for observables with multiple components, e.g. spin up and down.

    Components are to be represented by simple container classes (e.g. `obj`), 
    each having the same internal structure of the data.

    Class attributes
    ----------------
    component_names : `tuple`
        List of allowed components appearing as attributes in instances.
    default_component_name : `str`
        Default component name if no other is specified in a request.
    """

    component_names        = None
    default_component_name = None


    def process_component_name(self,name):
        if name is None:
            name = self.default_component_name
        elif name not in self.components:
            self.error('"{}" is not a known component.\nValid options are: {}'.format(name,self.component_names))
        #end if
        return name
    #end def process_component_name


    def default_component(self):
        """
        Return the default component
        """
        return self.component(self.default_component_name)
    #end def default_component


    def component(self,name):
        """
        Return a requested named component.
        """
        if name is None:
            return self.default_component()
        #end if
        if name not in self.component_names:
            self.error('"{}" is not a known component.\nValid options are: {}'.format(name,self.component_names))
        elif name not in self:
            self.error('Component "{}" not found.'.format(name))
        #end if
        comp = self.get_attribute(name)
        return comp
    #end def component


    def components(self,names=None):
        """
        Return all (or a subset of) the components.
        """
        comps = obj()
        if names is None:
            for c in self.component_names:
                if c in self:
                    comps[c] = self[c]
                #end if
            #end for
            if len(comps)==0:
                self.error('No components found.')
            #end if
        else:
            if isinstance(names,str):
                names = [names]
            #end if
            for name in names:
                if name not in self.component_names:
                    self.error('"{}" is not a known component.\nValid options are: {}'.format(name,self.component_names))
                elif name not in self:
                    self.error('Component "{}" not found.'.format(name))
                #end if
                comps[name] = self[name]
            #end for
        #end if
        return comps
    #end def components

#end class ObservableWithComponents

ObservableWithComponents.define_attributes(Observable)



def rinscribe(axes):
    radius = 1e99
    dim = len(axes)
    volume = abs(np.linalg.det(axes))
    for i in range(dim):
        j = (i+1)%dim
        rc = np.cross(axes[i,:],axes[j,:])
        radius = min(radius,.5*volume/np.linalg.norm(rc))
    #end for
    return radius
#end def rinscribe



def read_eshdf_nofk_data(filename,Ef):
    """
    Read n(k) data from an ESHDF file based on a provided Fermi energy.

    Parameters
    ----------
    Ef : `float`
        Fermi energy threshold in eV.  Orbitals with energies lower than 
        `Ef` are included to form the momentum distribution.
    """
    from numpy import array,pi,dot,sqrt,abs,zeros
    from numpy.linalg import inv,det
    from hdfreader import read_hdf

    def h5int(i):
        return array(i,dtype=int)[0]
    #end def h5int

    # Use slightly shifted Fermi energy
    E_fermi  = Ef + 1e-8

    # Open the HDF file w/o loading the arrays into memory (view mode)
    vlog('Reading '+filename)
    h        = read_hdf(filename,view=True)

    # Get the G-vectors in cell coordinates
    gvu      = array(h.electrons.kpoint_0.gvectors)

    # Get the untiled cell axes
    axes     = array(h.supercell.primitive_vectors)

    # Compute the k-space cell axes
    kaxes    = 2*pi*inv(axes).T

    # Convert G-vectors from cell coordinates to atomic units 
    gv       = dot(gvu,kaxes)

    # Get number of kpoints/twists, spins, and G-vectors
    nkpoints = h5int(h.electrons.number_of_kpoints)
    nspins   = h5int(h.electrons.number_of_spins)
    ngvecs   = len(gv)

    # Process the orbital data
    data     = obj()
    for k in range(nkpoints):
        vlog('Processing k-point {:>3}'.format(k),n=1,time=True)
        kin_k   = obj()
        eig_k   = obj()
        k_k     = obj()
        nk_k    = obj()
        nelec_k = zeros((nspins,),dtype=float)
        kp      = h.electrons['kpoint_'+str(k)]
        gvs     = dot(array(kp.reduced_k),kaxes)
        gvk     = gv.copy()
        for d in range(3):
            gvk[:,d] += gvs[d]
        #end for
        kinetic=(gvk**2).sum(1)/2 # Hartree units
        for s in range(nspins):
            kin_s   = []
            eig_s   = []
            k_s     = gvk
            nk_s    = zeros((ngvecs,),dtype=float)
            nelec_s = 0
            path    = 'electrons/kpoint_{0}/spin_{1}'.format(k,s)
            spin    = h.get_path(path)
            eigs    = convert(array(spin.eigenvalues),'Ha','eV')
            nstates = h5int(spin.number_of_states)
            for st in range(nstates):
                eig = eigs[st]
                if eig<E_fermi:
                    stpath   = path+'/state_{0}/psi_g'.format(st)
                    psi      = array(h.get_path(stpath))
                    nk_orb   = (psi**2).sum(1)
                    kin_orb  = (kinetic*nk_orb).sum()
                    nelec_s += nk_orb.sum()
                    nk_s    += nk_orb
                    kin_s.append(kin_orb)
                    eig_s.append(eig)
                #end if
            #end for
            data[k,s] = obj(
                kpoint = array(kp.reduced_k),
                kin    = array(kin_s),
                eig    = array(eig_s),
                k      = k_s,
                nk     = nk_s,
                ne     = nelec_s,
                )
        #end for
    #end for
    res = obj(
        orbfile  = filename,
        E_fermi  = E_fermi,
        axes     = axes,
        kaxes    = kaxes,
        nkpoints = nkpoints,
        nspins   = nspins,
        data     = data,
        )

    return res
#end def read_eshdf_nofk_data



class MomentumDistribution(ObservableWithComponents):
    """
    Momentum distribution analysis class.

    This class contains shared functionality for both deterministic (e.g. 
    DFT derived) and statistical (e.g. QMC derived) momentum distributions.
    Each of the processed up/down/up+down/up-down components are represented 
    by `GridFunction` objects.

    Attributes
    ----------
    raw : `obj`
        Container holding raw n(k) data prior to any filtering or grid mapping.
    u : `ParallelotopeGridFunction`
        Processed component data for the up spin channel.
    d : `ParallelotopeGridFunction`
        Processed component data for the down spin channel.
    tot : `ParallelotopeGridFunction`
        Processed component data for the up+down.
    pol : `ParallelotopeGridFunction`
        Processed component data for the up-down.
    kaxes : `ndarray`
        K-space axes of the simulation cell.  Stored in in `info`.
    raw_filter_tol : `float`
        Tolerance applied when filtering the raw data.  Stored in `info`.
    """

    component_names = ('tot','pol','u','d')
    
    default_component_name = 'tot'

    def get_raw_data(self):
        """
        Return raw data, checking that it has been initialized first.
        """
        data = self.get_attribute('raw')
        if len(data)==0:
            self.error('Raw n(k) data is not present.')
        #end if
        return data
    #end def get_raw_data


    def filter_raw_data(self,filter_tol=1e-5,store=True):
        """
        Filter out small values from the raw data.

        A maximum value of k-space radius is found such that no n(k) value 
        within the radius exceeds a tolerance.

        Parameters
        ----------
        filter_tol : `float, default 1e-5`
            Defines kmax = |km| such that n(k<km)>filter_tol.
        store : `bool, default True`
            Overwrite current raw data with truncated/filtered data.
        """
        vlog('Filtering raw n(k) data with tolerance {:6.4e}'.format(filter_tol))
        prior_tol = self.get_attribute('raw_filter_tol',assigned=False)
        data  = self.get_raw_data()
        if prior_tol is not None and prior_tol<=filter_tol:
            vlog('Filtering applied previously with tolerance {:6.4e}, skipping.'.format(prior_tol))
            return data
        #end if
        k     = data.first().k
        km    = np.linalg.norm(k,axis=1)
        kmax  = 0.
        order = km.argsort()
        for s,sdata in data.items():
            vlog('Finding kmax for {} data'.format(s),n=1,time=True)
            nk = sdata.nk
            for n in reversed(order):
                if nk[n]>filter_tol:
                    break
                #end if
            #end for
            kmax = max(km[n],kmax)
        #end for
        vlog('Original kmax: {:8.4f}'.format(km.max()),n=2)
        vlog('Filtered kmax: {:8.4f}'.format(kmax),n=2)
        vlog('Applying kmax filter to data',n=1,time=True)
        keep = km<kmax
        k = k[keep]
        vlog('size before filter: {}'.format(len(keep)),n=2)
        vlog('size  after filter: {}'.format(len(k)),n=2)
        vlog('fraction: {:6.4e}'.format(len(k)/len(keep)),n=2)
        if store:
            new_data = data
            self.set_attribute('raw_filter_tol',filter_tol)
        else:
            new_data = obj()
        #end if
        for s in data.keys():
            if s not in new_data:
                new_data[s] = obj()
            #end if
            sdata    = new_data[s]
            sdata.k  = k
            sdata.nk = data[s].nk[keep]
        #end for
        if store:
            vlog('Overwriting original raw n(k) with filtered data',n=1)
            self.set_attribute('raw',new_data)
        #end if
        vlog('Filtering complete',n=1,time=True)
        return new_data
    #end def filter_raw_data


    def map_raw_data_onto_grid(self,unfold=False,filter_tol=1e-5):
        """
        Initialize component data from raw data by creating rectilinear 
        bounding grids around all defined raw data values.

        Defined values may fall on a regularly spaced, but irregularly shaped 
        grid, e.g. cartesian points defined within a sphere.  In this case, 
        the rectilinear grid is padded with zeros around the irregular volume.
        This function also has the capability to unfold points by symmetry and 
        average over multi-valued points arising from invariance under some 
        symmetry operations.

        Parameters
        ----------
        unfold : `bool, default False`
            Use point group symmetries to unfold the data.  Requires presence 
            of `structure` attribute.
        filter_tol : `float, default=1e-5`
            Filter out n(k) data with values larger than `filter_tol`.
        """
        vlog('\nMapping raw n(k) data onto regular grid')
        data = self.get_raw_data()
        structure = self.get_attribute('structure',assigned=unfold)
        if structure is not None:
            kaxes = structure.kaxes
        else:
            kaxes = self.get_attribute('kaxes')
        #end if
        if filter_tol is not None:
            vlog.increment()
            data = self.filter_raw_data(filter_tol,store=False)
            vlog.decrement()
        #end if
        if not unfold:
            for s,sdata in data.items():
                vlog('Mapping {} data onto grid'.format(s),n=1,time=True)
                vlog.increment(2)
                gf = grid_function(
                    points        = sdata.k,
                    values        = sdata.nk,
                    axes          = kaxes,
                    check_compact = True,
                    )
                self.set_attribute(s,gf)
                vlog.decrement(2)
            #end for
        else:
            rotations = structure.point_group_operations()
            for s,sdata in data.items():
                if s=='d' and 'u' in data and id(sdata)==id(data.u):
                    continue
                #end if
                vlog('Unfolding {} data'.format(s),n=1,time=True)
                k   = []
                nk  = []
                ks  = sdata.k
                nks = sdata.nk
                for n,R in enumerate(rotations):
                    vlog('Processing rotation {:<3}'.format(n),n=2,mem=True)
                    k.extend(np.dot(ks,R))
                    nk.extend(nks)
                #end for
                k  = np.array(k ,dtype=float)
                nk = np.array(nk,dtype=float)
                vlog('Unfolding finished',n=2,time=True)

                vlog('Mapping {} data onto grid'.format(s),n=1,time=True)
                vlog.increment(2)
                gf = grid_function(
                    points        = k,
                    values        = nk,
                    axes          = kaxes,
                    average       = True, # Avg any multi-valued points
                    check_compact = True,
                    )
                self.set_attribute(s,gf)
                vlog.decrement(2)
            #end for
        #end if
        if 'd' not in self and 'u' in self:
            self.d = self.u
        #end if
        vlog('Mapping complete',n=1,time=True)
        vlog('Current memory: ',n=1,mem=True)
    #end def map_raw_data_onto_grid


    def backfold(self):
        """
        Fold extended zone n(k) data back into the reciprocal primitive cell.
        """
        # implementation needs more work
        self.not_implemented()

        structure = self.get_attribute('structure',assigned=True)
        kaxes     = structure.kaxes
        c         = self.default_component()
        dk        = c.grid.dr
        #print(kaxes)
        #print(dk)
        #print(np.diag(kaxes)/np.diag(dk))
        #print(c.grid.cell_grid_shape)
        #ci()
        #exit()
    #end def backfold


    def angular_average(self,component=None,dk=0.01,ntheta=100,kmax=None,single=False,interp_kwargs=None,comps_return=False):
        """
        Compute angular average of interpolated n(k).
        """
        vlog('Computing angular average',time=True)
        vlog('Current memory:',n=1,mem=True)
        if interp_kwargs is None:
            interp_kwargs = obj()
        #end if
        vlog('Constructing spherical grid',n=1,time=True)
        if kmax is None:
            c = self.default_component()
            kmax = rinscribe(c.grid.axes)
        #end if
        nkp = int(np.ceil(kmax/dk))
        sgrid = SpheroidGrid(
            axes     = kmax*np.eye(3),
            cells    = (nkp,ntheta,2*ntheta),
            centered = True,
            )
        nkrs = obj()
        for cname,nk in self.components(component).items():
            vlog('Processing angular average for component "{}"'.format(cname),n=1,time=True)
            nksphere = nk.interpolate(sgrid,**interp_kwargs)
            r,nkr = nksphere.angular_average()
            nkrs[cname] = obj(
                radius  = r,
                average = nkr,
                )
        #end for
        vlog('Current memory:',n=2,mem=True)
        return nkrs
    #end def angular_average


    def plot_plane_contours(self,
                            quantity     = None,
                            origin       = None,
                            a1           = None,
                            a2           = None,
                            a1_range     = (0,1),
                            a2_range     = (0,1),
                            grid_spacing = 0.3,
                            unit_in      = False,
                            unit_out     = False,
                            boundary     = True,
                            ):
        """
        Plot n(k) contours along a planar slice.
        """
        c  = self.component(quantity)
        o  = np.asarray(origin)
        a1 = np.asarray(a1)
        a2 = np.asarray(a2)
        if unit_in:
            s = self.get_attribute('structure',assigned=True)
            from structure import get_seekpath_full
            skp   = get_seekpath_full(structure=s,primitive=True)
            kaxes = np.asarray(skp.reciprocal_primitive_lattice)
            o_in  = o
            a1_in = a1
            a2_in = a2
            o     = np.dot(o_in ,kaxes)
            a1    = np.dot(a1_in,kaxes)
            a2    = np.dot(a2_in,kaxes)
            special_kpoints = skp.point_coords
        #end if
        a1    -= o
        a2    -= o
        corner = o + a1_range[0]*a1 + a2_range[0]*a2
        a1    *= a1_range[1] - a1_range[0]
        a2    *= a2_range[1] - a2_range[0]
        g = generate_grid(
            type   = 'parallelotope',
            corner = corner,
            axes   = [a1,a2],
            dr     = (grid_spacing,grid_spacing),
            )
        gf = c.interpolate(g)
        gf.plot_contours(boundary=boundary)
    #end def plot_plane_contours


    def plot_radial_average(self,quants='all',kmax=None,fmt='b-',fig=True,show=True,interp_kwargs=None):
        """
        Plot radial n(k) following angular average.
        """
        if quants=='all':
            quants = list(data.keys())
        #end if
        nkrs = self.angular_average(component=quants,kmax=kmax,interp_kwargs=interp_kwargs)
        for q in quants:
            nk = nkrs[q]
            if fig:
                plt.figure()
            #end if
            plt.plot(nk.radius,nk.average,fmt)
            plt.xlabel('k (a.u.)')
            plt.ylabel('n(k) {}'.format(q))
        #end if
        if show:
            plt.show()
        #end if
    #end def plot_radial_average


    def plot_radial_scatter(self,quants='all',kmax=None,fmt='b.',fig=True,show=True,grid=True,raw=False,raw_fmt=None):
        """
        Plot all n(k) data points as a function of k-space radius.
        """
        data = self.get_raw_data()
        if quants=='all':
            quants = list(data.keys())
        #end if
        source_formats = []
        if grid:
            source_formats.append(('grid',fmt))
        #end if
        if raw:
            if raw_fmt is not None:
                rfmt = raw_fmt
            elif grid:
                rfmt = 'rx'
            else:
                rfmt = fmt
            #end if
            source_formats.append(('raw',rfmt))
        #end if
        for q in quants:
            if fig:
                plt.figure()
            #end if
            for source,fmt in source_formats:
                if source=='grid':
                    d  = self.component(q)
                    k  = d.r
                    nk = d.f
                elif source=='raw':
                    d  = data[q]
                    k  = d.k
                    nk = d.nk
                #end if
                k  = np.linalg.norm(k,axis=1)
                has_error = 'nk_err' in d
                if has_error:
                    nke = d.nk_err
                #end if
                if kmax is not None:
                    rng = k<kmax
                    k   = k[rng]
                    nk  = nk[rng]
                    if has_error:
                        nke = nke[rng]
                    #end if
                #end if
                if not has_error:
                    plt.plot(k,nk,fmt)
                else:
                    plt.errorbar(k,nk,nke,fmt=fmt)
                #end if
            #end for
            plt.xlabel('k (a.u.)')
            plt.ylabel('n(k) {}'.format(q))
        #end for
        if show:
            plt.show()
        #end if
    #end def plot_radial_scatter


    def plot_directional(self,kdir,quants='all',kmax=None,fmt='b.',fig=True,show=True,reflect=False,grid=True,raw=False,raw_fmt=None):
        """
        Plot all n(k) data points along a k-space direction.
        """
        data = self.get_raw_data()
        kdir = np.array(kdir,dtype=float)
        kdir /= np.linalg.norm(kdir)
        if quants=='all':
            quants = list(data.keys())
        #end if
        source_formats = []
        if grid:
            source_formats.append(('grid',fmt))
        #end if
        if raw:
            if raw_fmt is not None:
                rfmt = raw_fmt
            elif grid:
                rfmt = 'rx'
            else:
                rfmt = fmt
            #end if
            source_formats.append(('raw',rfmt))
        #end if
        for q in quants:
            if fig:
                plt.figure()
            #end if
            for source,fmt in source_formats:
                if source=='grid':
                    d  = self.component(q)
                    k  = d.r
                    nk = d.f
                elif source=='raw':
                    d  = data[q]
                    k  = d.k
                    nk = d.nk
                #end if
                has_error = 'nk_err' in d
                if has_error:
                    nke = d.nk_err
                #end if
                km = np.linalg.norm(k,axis=1)
                if kmax is not None:
                    rng = km<kmax
                    km  = km[rng]
                    k   = k[rng]
                    nk  = nk[rng]
                    if has_error:
                        nke = nke[rng]
                    #end if
                #end if
                kd = np.dot(k,kdir)
                along_dir = (np.abs(km-np.abs(kd)) < 1e-8*km) | (km<1e-8)
                kd = kd[along_dir]
                nk = nk[along_dir]
                if has_error:
                    nke = nke[along_dir]
                #end if
                if not has_error:
                    plt.plot(kd,nk,fmt)
                    if reflect:
                        plt.plot(-kd,nk,fmt)
                    #end if
                else:
                    plt.errorbar(kd,nk,nke,fmt=fmt)
                    if reflect:
                        plt.errorbar(-kd,nk,nke,fmt=fmt)
                    #end if
                #end if
            #end for
            plt.xlabel('k (a.u.)')
            plt.ylabel('directional n(k) {}'.format(q))
        #end for
    #end def plot_directional
#end class MomentumDistribution

MomentumDistribution.define_attributes(
    ObservableWithComponents,
    raw = obj(
        type       = obj,
        no_default = True,
        ),
    u = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    d = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    tot = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    pol = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    kaxes = obj(
        dest       = 'info',
        type       = np.ndarray,
        no_default = True,
        ),
    raw_filter_tol = obj(
        dest       = 'info',
        type       = float,
        default    = None,
        ),
    )



class MomentumDistributionDFT(MomentumDistribution):
    """
    Momentum distribution class for DFT data.

    Attributes
    ----------
    E_fermi : `float`
        Fermi energy (eV) defining the occupied orbitals.
    """

    def read_eshdf(self,filepath,E_fermi=None,savefile=None,tiling=None,unfold=False,grid=True):
        """
        Read raw n(k) data from an ESHDF file.
        """

        save = False
        if savefile is not None:
            if os.path.exists(savefile):
                vlog('\nLoading from save file {}'.format(savefile))
                self.load(savefile)
                vlog('Done',n=1,time=True)
                return
            else:
                save = True
            #end if            
        #end if
                
        vlog('\nExtracting n(k) data from {}'.format(filepath))

        if E_fermi is None:
            E_fermi = self.info.E_fermi
        else:
            self.info.E_fermi = E_fermi
        #end if
        if E_fermi is None:
            self.error('Cannot read n(k) from ESHDF file.  Fermi energy (eV) is required to populate n(k) from ESHDF data.\nFile being read: {}'.format(filepath))
        #end if

        vlog.increment()
        d = read_eshdf_nofk_data(filepath,E_fermi)
        vlog.decrement()

        spins = {0:'u',1:'d'}

        spin_data = obj()
        for (ki,si) in sorted(d.data.keys()):
            vlog('Appending data for k-point {:>3} and spin {}'.format(ki,si),n=1,time=True)
            data = d.data[ki,si]
            s = spins[si]
            if s not in spin_data:
                spin_data[s] = obj(k=[],nk=[])
            #end if
            sdata = spin_data[s]
            sdata.k.extend(data.k)
            sdata.nk.extend(data.nk)
        #end for
        for sdata in spin_data:
            sdata.k  = np.array(sdata.k)
            sdata.nk = np.array(sdata.nk)
        #end for
        if 'd' not in spin_data:
            spin_data.d = spin_data.u
        #end if
        spin_data.tot = obj(
            k  = spin_data.u.k,
            nk = spin_data.u.nk + spin_data.d.nk,
            )

        kaxes = d.kaxes
        if tiling is not None:
            tiling = np.array(tiling,dtype=float)
            if tiling.size==3:
                tiling = np.diag(tiling)
            #end if
            tiling.shape = (3,3)
            kaxes = np.dot(np.linalg.inv(tiling.T),kaxes)
        #end if

        self.set_attribute('raw'  , spin_data)
        self.set_attribute('kaxes', kaxes    )

        if grid:
            self.map_raw_data_onto_grid(unfold=unfold)
        #end if

        if save:
            vlog('Saving to file {}'.format(savefile),n=1)
            self.save(savefile)
        #end if

        vlog('n(k) data extraction complete',n=1,time=True)

    #end def read_eshdf
#end class MomentumDistributionDFT

MomentumDistributionDFT.define_attributes(
    MomentumDistribution,
    E_fermi = obj(
        dest    = 'info',
        type    = float,
        default = None,
        )
    )


class MomentumDistributionQMC(MomentumDistribution):
    """
    Momentum distribution class for QMC data.
    """

    def read_stat_h5(self,*files,equil=0,savefile=None):
        """
        Read raw n(k) data from QMCPACK stat.h5 files.

        Data is roughly postprocessed to obtain means and errorbars.
        Data from multiple files (e.g. from different twists) is simply 
        appended for later postprocessing.
        """
        save = False
        if savefile is not None:
            if os.path.exists(savefile):
                vlog('\nLoading from save file {}'.format(savefile))
                self.load(savefile)
                vlog('Done',n=1,time=True)
                return
            else:
                save = True
            #end if            
        #end if

        vlog('\nReading n(k) data from stat.h5 files',time=True)
        k   = []
        nk  = []
        nke = []
        if len(files)==1 and isinstance(files[0],(list,tuple)):
            files = files[0]
        #end if
        for file in files:
            if isinstance(file,StatFile):
                stat = file
            else:
                vlog('Reading stat.h5 file',n=1,time=True)
                stat = StatFile(file,observables=['momentum_distribution'])
            #end if
            vlog('Processing n(k) data from stat.h5 file',n=1,time=True)
            vlog('filename = {}'.format(stat.filepath),n=2)
            group = stat.observable_groups(self,single=True)

            kpoints = np.array(group['kpoints'])
            nofk    = np.array(group['value'])

            nk_mean,nk_var,nk_error,nk_kappa = simstats(nofk[equil:],dim=0)

            k.extend(kpoints)
            nk.extend(nk_mean)
            nke.extend(nk_error)
        #end for
        vlog('Converting concatenated lists to arrays',n=1,time=True)
        data = obj(
            tot = obj(
                k      = np.array(k),
                nk     = np.array(nk),
                nk_err = np.array(nke),
                )
            )
        self.set_attribute('raw',data)

        if save:
            vlog('Saving to file {}'.format(savefile),n=1)
            self.save(savefile)
        #end if

        vlog('stat.h5 file read finished',n=1,time=True)
    #end def read_stat_h5
#end class MomentumDistributionQMC



class Density(ObservableWithComponents):
    """
    Density analysis class.

    Attributes
    ----------
    raw : `obj`
        Container holding raw density data prior to any filtering or grid mapping.
    u : `ParallelotopeGridFunction`
        Processed component data for the up spin channel.
    d : `ParallelotopeGridFunction`
        Processed component data for the down spin channel.
    tot : `ParallelotopeGridFunction`
        Processed component data for the up+down.
    pol : `ParallelotopeGridFunction`
        Processed component data for the up-down.
    grid : `ParallelotopeGrid`
        Grid on which all density components are defined.
    distance_units : `str`
        Spatial distance units (e.g. A=Angstrom, B=Bohr).
    density_units : `str`
        Density units.
    """

    component_names = ('tot','pol','u','d')

    default_component_name = 'tot'


    def read_xsf(self,filepath,component=None):
        """
        Read density data for a particular component from an XSF file.
        """
        component = self.process_component_name(component)

        vlog('Reading density data from XSF file for component "{}"'.format(component),time=True)

        if isinstance(filepath,XsfFile):
            vlog('XSF file already loaded, reusing data.')
            xsf = filepath
            copy_values = True
        else:
            vlog('Loading data from file',n=1,time=True)
            vlog('file location: {}'.format(filepath),n=2)
            vlog('memory before: ',n=2,mem=True)
            xsf = XsfFile(filepath)
            vlog('load complete',n=2,time=True)
            vlog('memory after: ',n=2,mem=True)
            copy_values = False
        #end if

        # read structure
        if not self.has_attribute('structure'):
            vlog('Reading structure from XSF data',n=1,time=True)
            s = Structure()
            s.read_xsf(xsf)
            self.set_attribute('structure',s)
        #end if

        # read grid
        if not self.has_attribute('grid'):
            vlog('Reading grid from XSF data',n=1,time=True)
            g = read_grid(xsf)
            self.set_attribute('grid',g)
            self.set_attribute('distance_units','B')
        #end if

        # read values
        xsf.remove_ghost()
        d = xsf.get_density()
        values = d.values_noghost.ravel()
        if copy_values:
            values = values.copy()
        #end if

        # create grid function for component
        vlog('Constructing grid function from XSF data',n=1,time=True)
        f = grid_function(
            type   = 'parallelotope',
            grid   = self.grid,
            values = values,
            copy   = False,
            )

        self.set_attribute(component,f)
        self.set_attribute('distance_units','A')

        vlog('Read complete',n=1,time=True)
        vlog('Current memory:',n=1,mem=True)
    #end def read_xsf

    
    def volume_normalize(self):
        """
        Normalize data by volume so that volume integrals converge to electron counts.
        """
        g = self.get_attribute('grid')
        dV = g.volume()/g.ncells
        for c in self.components():
            c.values /= dV
        #end for
    #end def volume_normalize


    def norm(self,component=None):
        """
        Perform volume integrals of all density components.
        """
        norms = obj()
        comps = self.components(component)
        for name,d in comps.items():
            g = d.grid
            dV = g.volume()/g.ncells
            norms[name] = d.values.sum()*dV
        #end if
        if isinstance(component,str):
            return norms[component]
        else:
            return norms
        #end if
    #end def norm


    def change_distance_units(self,units):
        """
        Change internal distance units.
        """
        units_old = self.get_attribute('distance_units')
        rscale    = 1.0/convert(1.0,units_old,units)
        dscale    = 1./rscale**3
        grid      = self.get_attribute('grid')
        grid.points *= rscale
        for c in self.components():
            c.values *= dscale
        #end for
    #end def change_distance_units


    def change_density_units(self,units):
        """
        Change internal density units.
        """
        units_old = self.get_attribute('density_units')
        dscale    = 1.0/convert(1.0,units_old,units)
        for c in self.components():
            c.values *= dscale
        #end for
    #end def change_density_units


    def radial_density(self,component=None,dr=0.01,ntheta=100,rmax=None,single=False,interp_kwargs=None,comps_return=False):
        """
        Compute radial density profiles around selected atoms.

        Capable of identifying and averaging over sets of symmetry equivalent atoms. 
        """
        vlog('Computing radial density',time=True)
        vlog('Current memory:',n=1,mem=True)
        if interp_kwargs is None:
            interp_kwargs = obj()
        #end if
        s = self.get_attribute('structure')
        struct = s
        if rmax is None:
            rmax = s.voronoi_species_radii()
        #end if
        vlog('Finding equivalent atomic sites',n=1,time=True)
        equiv_atoms = s.equivalent_atoms()
        species = None
        species_rmax = obj()
        if isinstance(rmax,float):
            species = list(equiv_atoms.keys())
            for s in species:
                species_rmax[s] = rmax
            #end for
        else:
            species = list(rmax.keys())
            species_rmax.transfer_from(rmax)
        #end if
        vlog('Constructing spherical grid for each species',n=1,time=True)
        species_grids = obj()
        for s in species:
            srmax = species_rmax[s]
            if srmax<1e-3:
                self.error('Cannot compute radial density.\n"rmax" must be set to a finite value.\nrmax provided for species "{}": {}'.format(s,srmax))
            #end if
            nr = int(np.ceil(srmax/dr))
            species_grids[s] = SpheroidGrid(
                axes     = srmax*np.eye(3),
                cells    = (nr,ntheta,2*ntheta),
                centered = True,
                )
        #end for

        rdfs = obj()
        for cname,d in self.components(component).items():
            vlog('Processing radial density for component "{}"'.format(cname),n=1,time=True)
            rdf = obj()
            rdfs[cname] = rdf
            for s,sgrid in species_grids.items():
                rrad    = sgrid.radii()
                rsphere = sgrid.r
                drad    = np.zeros(rrad.shape,dtype=d.dtype)
                if single:
                    atom_indices = [equiv_atoms[s][0]]
                else:
                    atom_indices = equiv_atoms[s]
                #end if
                vlog('Averaging radial data for species "{}" over {} sites'.format(s,len(atom_indices)),n=2,time=True)
                rcenter = np.zeros((3,),dtype=float)
                for i in atom_indices:
                    new_center = struct.pos[i]
                    dr         = new_center-rcenter
                    rsphere   += dr
                    rcenter    = new_center
                    dsphere    = d.interpolate(rsphere,**interp_kwargs)
                    dsphere.shape = sgrid.shape
                    dsphere.shape = len(dsphere),dsphere.size//len(dsphere)
                    drad += dsphere.mean(axis=1)*4*np.pi*rrad**2
                #end for
                drad /= len(atom_indices)
                rdf[s] = obj(
                    radius  = rrad,
                    density = drad,
                    )
            #end for
            d.clear_ghost()
            vlog('Current memory:',n=2,mem=True)
        #end if
        if isinstance(component,str) and not comps_return:
            return rdfs[component]
        else:
            return rdfs
        #end if
    #end def radial_density


    def cumulative_radial_density(self,rdfs=None,comps_return=False,**kwargs):
        """
        Compute cumulative radial density profiles around selected atoms.
        """
        component = kwargs.get('component',None)
        if rdfs is None:
            kwargs['comps_return'] = True
            crdfs = self.radial_density(**kwargs)
        else:
            crdfs = rdfs.copy()
        #end if
        for crdf in crdfs:
            for d in crdf:
                dr = d.radius[1]-d.radius[0]
                d.density = d.density.cumsum()*dr
            #end for
        #end if
        if isinstance(component,str) and not comps_return:
            return crdfs[component]
        else:
            return crdfs
        #end if
    #end def cumulative_radial_density


    def plot_radial_density(self,component=None,show=True,cumulative=False,**kwargs):
        """
        Plot radial density profiles around selected atoms.
        """
        vlog('Plotting radial density')
        kwargs['comps_return'] = True
        if not cumulative:
            rdfs = self.radial_density(component=component,**kwargs)
        else:
            rdfs = self.cumulative_radial_density(component=component,**kwargs)
        #end if
        rdf = rdfs.first()
        species = list(rdf.keys())

        dist_units = self.get_attribute('distance_units',None)

        for cname in self.component_names:
            if cname in rdfs:
                rdf = rdfs[cname]
                for s in sorted(rdf.keys()):
                    srdf = rdf[s]
                    plt.figure()
                    plt.plot(srdf.radius,srdf.density,'b.-')
                    xlabel = 'Radius'
                    if dist_units is not None:
                        xlabel += ' ({})'.format(dist_units)
                    #end if
                    plt.xlabel(xlabel)
                    if not cumulative:
                        plt.ylabel('Radial density')
                    else:
                        plt.ylabel('Cumulative radial density')
                    #end if
                    plt.title('{} {} density'.format(s,cname))
                #end for
            #end if
        #end for
        if show:
            plt.show()
        #end if
    #end def plot_radial_density


    def save_radial_density(self,prefix,rdfs=None,**kwargs):
        """
        Save radial density profiles for selected atoms.
        """
        path = ''
        if '/' in prefix:
            path,prefix = os.path.split(prefix)
        #end if
        vlog('Saving radial density with file prefix "{}"'.format(prefix))
        vlog.increment()
        kwargs['comps_return'] = True
        if rdfs is None:
            rdfs = self.radial_density(**kwargs)
        #end if
        crdfs = self.cumulative_radial_density(rdfs)
        vlog.decrement()
        groups = obj(
            rad_dens     = rdfs,
            rad_dens_cum = crdfs,
            )
        for gname,dfs in groups.items():
            for cname,rdf in dfs.items():
                for sname,srdf in rdf.items():
                    filename = '{}.{}.{}_{}.dat'.format(prefix,gname,sname,cname)
                    filepath = os.path.join(path,filename)
                    vlog('Saving file '+filepath,n=1)
                    f = open(filepath,'w')
                    for r,d in zip(srdf.radius,srdf.density):
                        f.write('{: 16.8e} {: 16.8e}\n'.format(r,d))
                    #end for
                    f.close()
                #end for
            #end for
        #end for
    #end def save_radial_density
#end class Density

Density.define_attributes(
    Observable,
    raw = obj(
        type       = obj,
        no_default = True,
        ),
    u = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    d = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    tot = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    pol = obj(
        type       = ParallelotopeGridFunction,
        no_default = True,
        ),
    grid = obj(
        type       = ParallelotopeGrid,
        no_default = True,
        ),
    distance_units = obj(
        dest       = 'info',
        type       = str,
        ),
    density_units = obj(
        dest      = 'info',
        type      = str,
        )
    )



class ChargeDensity(Density):
    None
#end class ChargeDensity


class EnergyDensity(Density):
    None
#end class EnergyDensity




class StatFile(DevBase):
    """
    Class to organize and extract raw observable data from QMCPACK's
    stat.h5 files.
    """

    scalars = set('''
        LocalEnergy   
        LocalEnergy_sq
        Kinetic       
        LocalPotential
        ElecElec      
        IonIon        
        LocalECP      
        NonLocalECP   
        KEcorr        
        MPC           
        '''.split())

    observable_aliases = obj(
        momentum_distribution = ['nofk'],
        )
    for observable in list(observable_aliases.keys()):
        for alias in observable_aliases[observable]:
            observable_aliases[alias] = observable
        #end for
        observable_aliases[observable] = observable
    #end for

    observable_classes = obj(
        momentum_distribution = MomentumDistributionQMC,
        )

    observable_class_to_stat_group = obj()
    for name,cls in observable_classes.items():
        observable_class_to_stat_group[cls.__name__] = name
    #end for


    def __init__(self,filepath=None,**read_kwargs):
        self.filepath = None

        if filepath is not None:
            self.filepath = filepath
            self.read(filepath,**read_kwargs)
        #end if
    #end def __init__

            
    def read(self,filepath,observables='all'):
        if not os.path.exists(filepath):
            self.error('Cannot read file.\nFile path does not exist: {}'.format(filepath))
        #end if
        h5 = h5py.File(filepath,'r')
        observable_groups = obj()
        for name,group in h5.items():
            # Skip scalar quantities
            if name in self.scalars:
                continue
            #end if
            # Identify observable type by name, for now
            for alias,observable in self.observable_aliases.items():
                cond_name  = self.condense_name(name)
                cond_alias = self.condense_name(alias)
                if cond_name.startswith(cond_alias):
                    if observable not in observable_groups:
                        observable_groups[observable] = obj()
                    #end if
                    observable_groups[observable][name] = group
                #end if
            #end for
        #end for
        if isinstance(observables,str):
            if observables=='all':
                self.transfer_from(observable_groups)
            #end if
        else:
            for obs in observables:
                if obs in observable_groups:
                    self[obs] = observable_groups[obs]
                #end if
            #end for
        #end if
    #end def read


    def condense_name(self,name):
        return name.lower().replace('_','')
    #end def condenst_name


    def observable_groups(self,observable,single=False):
        if inspect.isclass(observable):
            observable = observable.__name__
        elif isinstance(observable,Observable):
            observable = observable.__class__.__name__
        #end if
        groups = None
        if observable in self:
            groups = self[observable]
        elif observable in self.observable_class_to_stat_group:
            observable = self.observable_class_to_stat_group[observable]
            if observable in self:
                groups = self[observable]
            #end if
        #end if
        if single and groups is not None:
            if len(groups)==1:
                return groups.first()
            else:
                self.error('Single stat.h5 observable group requested, but multiple are present.\nGroups present: {}'.format(sorted(groups.keys())))
            #end if
        else:
            return groups
        #end if
    #end def observable_groups
#end class StatFile
