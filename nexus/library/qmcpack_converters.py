##################################################################
##  (c) Copyright 2015-  by Jaron T. Krogel                     ##
##################################################################


#====================================================================#
#  qmcpack_converters.py                                             #
#    Nexus interfaces for orbital converter tools of QMCPACK:        #
#    pw2qmcpack, wfconvert, and convert4qmc                          #
#                                                                    #
#  Content summary:                                                  #
#    generate_pw2qmcpack                                             #
#      User-facing function to create pw2qmcpack simulation objects. #
#                                                                    #
#    generate_pw2qmcpack_input                                       #
#      User-facing funcion to create input for pw2qmcpack.           #
#                                                                    #
#    Pw2qmcpack                                                      #
#      Simulation class for pw2qmcpack.                              #
#                                                                    #
#    Pw2qmcpackInput                                                 #
#      SimulationInput class for pw2qmcpack.                         #
#                                                                    #
#    Pw2qmcpackAnalyzer                                              #
#      SimulationAnalyzer class for pw2qmcpack.                      #
#                                                                    #
#                                                                    #
#    WfconvertInput                                                  #
#      SimulationInput class for wfconvert.                          #     
#                                                                    #
#    WfconvertAnalyzer                                               #
#      SimulationAnalyzer class for wfconvert.                       #        
#                                                                    #
#    Wfconvert                                                       #
#      Simulation class for wfconvert.                               #
#                                                                    #
#    generate_wfconvert                                              #
#      User-facing function to generate wfconvert simulation objects.#
#                                                                    #
#                                                                    #
#    Convert4qmcInput                                                #
#      Class representing command line interface of convert4qmc.     #
#                                                                    #
#    Convert4qmcAnalyzer                                             #
#      Placeholder class for output analysis.                        #
#                                                                    #
#    Convert4qmc                                                     #
#      Class representing convert4qmc instance.                      #
#                                                                    #
#    generate_convert4qmc_input                                      #
#      Function to generate arbitrary convert4qmc input.             #
#                                                                    #
#    generate_convert4qmc                                            #
#      Function to generate Convert4qmc simulation object.           #
#                                                                    #
#====================================================================#



import os
from numpy import array,zeros
from generic import obj
from simulation import Simulation,SimulationInput,SimulationAnalyzer
from gamess import Gamess
from fileio import TextFile

# read/write functions associated with pw2qmcpack only
def read_str(sv):
    return sv.strip('"').strip("'")
#end def read_str

def read_int(sv):
    return int(sv)
#end def read_int

def read_float(sv):
    return float(sv.replace('d','e').replace('D','e'))
#end def read_float

bconv = {'.true.':True,'.false.':False}
def read_bool(sv):
    return bconv[sv]
#end def read_bool

def write_str(val):
    return "'"+val+"'"
#end def write_str

def write_int(val):
    return str(val)
#end def write_int

def write_float(val):
    return str(val)
#end def write_float

def write_bool(val):
    return '.'+str(val).lower()+'.'
#end def write_bool

readval={str:read_str,int:read_int,float:read_float,bool:read_bool}
writeval={str:write_str,int:write_int,float:write_float,bool:write_bool}


class Pw2qmcpackInput(SimulationInput):
    ints   = []
    floats = []
    strs   = ['outdir','prefix']
    bools  = ['write_psir']

    var_types = dict()
    for v in ints:
        var_types[v]=int
    for v in floats:
        var_types[v]=float
    for v in strs:
        var_types[v]=str
    for v in bools:
        var_types[v]=bool

    allowed = set(ints+floats+strs+bools)

    def read_text(self,contents,filepath=None):
        lines = contents.split('\n')
        inside = False
        for l in lines:
            if inside:
                tokens = l.split(',')
                for t in tokens:
                    ts = t.strip()
                    if ts!='' and ts!='/':
                        name,value = ts.split('=')
                        name = name.strip()
                        value= value.strip()
                        if name in self.allowed:
                            vtype = self.var_types[name]
                            value = readval[vtype](value)
                            sobj[name]=value
                        else:
                            self.error('encountered unknown variable: '+name)
                        #end if
                    #end if
                #end for
            #end if
            if '&' in l:
                inside=True
                section = l.lstrip('&').lower()
                sobj = obj()
                self[section]=sobj
            elif l.strip()=='/':
                inside=False
            #end if
        #end for
    #end def read_text

    def write_text(self,filepath=None):
        contents = ''
        for sname,section in self.iteritems():
            contents+='&'+sname+'\n'
            for name,value in section.iteritems():
                vtype = type(value)
                contents += '  '+name+' = '+writeval[vtype](value)+'\n'
            #end for
            contents+='/\n'
        #end for
        return contents
    #end def write_text


    def __init__(self,filepath=None,**vars):
        if filepath!=None:
            self.read(filepath)
        else:
            inputpp = obj()
            for name,value in vars.iteritems():
                inputpp[name] = value
            #end for
            self.inputpp = inputpp
        #end if
    #end def __init__
#end class Pw2qmcpackInput


def generate_pw2qmcpack_input(prefix='pwscf',outdir='pwscf_output',write_psir=False):
    pw = Pw2qmcpackInput(
        prefix     = prefix,
        outdir     = outdir,
        write_psir = write_psir
        )
    return pw
#end def generate_pw2qmcpack_input



class Pw2qmcpackAnalyzer(SimulationAnalyzer):
    def __init__(self,arg0):
        if isinstance(arg0,Simulation):
            sim = arg0
            self.infile = sim.infile
            prefix,outdir = sim.input.inputpp.tuple('prefix','outdir')
            self.dir = sim.locdir
            self.h5file = os.path.join(sim.locdir,outdir,prefix+'.pwscf.h5')
        else:
            self.infile = arg0
        #end if
    #end def __init__

    def analyze(self):
        if False:
            import h5py
            self.log('Fixing h5 file',n=5)

            path = os.path.split(self.h5file)[0]
            print os.getcwd()
            print os.listdir('./')
            if os.path.exists(path):
                print os.listdir(path)
            #end if
            print self.h5file

            h = h5py.File(self.h5file)
            if 'electrons' in h:
                elec = h['electrons']
                nkpoints = 0
                for name,val in elec.iteritems():
                    if name.startswith('kpoint'):
                        nkpoints+=1
                    #end for
                #end if
                nkold = elec['number_of_kpoints'][0] 
                self.log('Were',nkold,'kpoints, now',nkpoints,'kpoints',n=6)
                elec['number_of_kpoints'][0] = nkpoints
            #end for        
        #end if
    #end def analyze

    def get_result(self,result_name):
        self.not_implemented()
    #end def get_result
#end class Pw2qmcpackAnalyzer


class Pw2qmcpack(Simulation):
    input_type = Pw2qmcpackInput
    analyzer_type = Pw2qmcpackAnalyzer
    generic_identifier = 'pw2qmcpack'
    application = 'pw2qmcpack.x'
    application_properties = set(['serial'])
    application_results    = set(['orbitals'])

    def check_result(self,result_name,sim):
        calculating_result = False
        inputpp = self.input.inputpp
        if result_name=='orbitals':
            calculating_result = True
        else:
            calculating_result = False
            self.error('ability to check for result '+result_name+' has not been implemented')
        #end if        
        return calculating_result
    #end def check_result

    def get_result(self,result_name,sim):
        result = obj()
        inputpp = self.input.inputpp
        prefix = 'pwscf'
        outdir = './'
        if 'prefix' in inputpp:
            prefix = inputpp.prefix
        #end if
        if 'outdir' in inputpp:
            outdir = inputpp.outdir
        #end if
        if outdir.startswith('./'):
            outdir = outdir[2:]
        #end if
        if result_name=='orbitals':
            result.h5file   = os.path.join(self.locdir,outdir,prefix+'.pwscf.h5')
            result.ptcl_xml = os.path.join(self.locdir,outdir,prefix+'.ptcl.xml')
            result.wfs_xml  = os.path.join(self.locdir,outdir,prefix+'.wfs.xml')
        else:
            self.error('ability to get result '+result_name+' has not been implemented')
        #end if        
        return result
    #end def get_result

    def incorporate_result(self,result_name,result,sim):
        if result_name=='orbitals':
            pwin = sim.input.control
            p2in = self.input.inputpp
            pwprefix = 'pwscf'
            p2prefix = 'pwscf'
            pwoutdir = './'
            p2outdir = './'
            if 'prefix' in pwin:
                pwprefix = pwin.prefix
            if 'prefix' in p2in:
                p2prefix = p2in.prefix
            if 'outdir' in pwin:
                pwoutdir = pwin.outdir
            if 'outdir' in p2in:
                p2outdir = p2in.outdir
            if pwoutdir.startswith('./'):
                pwoutdir = pwoutdir[2:]
            if p2outdir.startswith('./'):
                p2outdir = p2outdir[2:]
            pwdir = os.path.abspath(os.path.join(sim.locdir ,pwoutdir))
            p2dir = os.path.abspath(os.path.join(self.locdir,p2outdir))
            errors = False
            if pwdir!=p2dir:
                self.error('to use orbitals, '+self.generic_identifier+' must have the same outdir as pwscf\n  pwscf outdir: '+pwdir+'\n  '+self.generic_identifier+' outdir: '+p2dir,exit=False)
                errors = True
            #end if
            if pwprefix!=p2prefix:
                self.error('to use orbitals, '+self.generic_identifier+' must have the same prefix as pwscf\n  pwscf prefix: '+pwprefix+'\n  '+self.generic_identifier+' prefix: '+p2prefix,exit=False)
                errors = True
            #end if
            if errors:
                self.error(self.generic_identifier+' cannot use orbitals from pwscf')
            #end if
        else:
            self.error('ability to incorporate result '+result_name+' has not been implemented')
        #end if                
    #end def incorporate_result

    def check_sim_status(self):
        outfile = os.path.join(self.locdir,self.outfile)
        fobj = open(outfile,'r')
        output = fobj.read()
        fobj.close()
        inputpp = self.input.inputpp
        prefix = 'pwscf'
        outdir = './'
        if 'prefix' in inputpp:
            prefix = inputpp.prefix
        #end if
        if 'outdir' in inputpp:
            outdir = inputpp.outdir
        #end if
        if outdir.startswith('./'):
            outdir = outdir[2:]
        #end if
        h5file   = os.path.join(self.locdir,outdir,prefix+'.pwscf.h5')
        ptcl_xml = os.path.join(self.locdir,outdir,prefix+'.ptcl.xml')
        wfs_xml  = os.path.join(self.locdir,outdir,prefix+'.wfs.xml')
        must_exist = [h5file,ptcl_xml,wfs_xml]

        files_exist = True
        for file in must_exist:
            files_exist = files_exist and os.path.exists(file)
        #end for
        outfin = True
        #outfin = outfin and 'esh5 create' in output
        #outfin = outfin and 'Creating electrons' in output
        outfin = outfin and 'npw=' in output
        outfin = outfin and 'ik=' in output

        outfin = outfin or 'JOB DONE' in output

        success = files_exist and outfin

        #self.finished = success and self.job.finished

        # pw2qmcpack has too many variants to assess completion based on log output
        #   assume (optimistically) that job completion indicates success
        self.finished = files_exist and self.job.finished
    #end def check_sim_status

    def get_output_files(self):
        output_files = []
        return output_files
    #end def get_output_files

    def app_command(self):
        return self.app_name+'<'+self.infile
    #end def app_command
#end class Pw2qmcpack




def generate_pw2qmcpack(**kwargs):
    sim_args,inp_args = Simulation.separate_inputs(kwargs)

    if not 'input' in sim_args:
        sim_args.input = generate_pw2qmcpack_input(**inp_args)
    #end if
    pw2qmcpack = Pw2qmcpack(**sim_args)

    return pw2qmcpack
#end def generate_pw2qmcpack















class WfconvertInput(SimulationInput):
    def __init__(self,app_name='wfconvert',h5in='MISSING.h5',h5out='wfconvert.h5',spline=False,format='eshdf',factor=None):
        self.app_name = app_name
        self.h5in = h5in
        self.h5out= h5out
        self.spline = spline
        self.format = format
        self.factor = factor
    #end def __init__

#wfconvert --nospline --eshdf diamond.h5 out/diamond.pwscf.h5 >& diamond-wfconvert.out 
    def set_app_name(self,app_name):
        self.app_name = app_name
    #end def set_app_name

    def app_command(self):
        c = self.app_name+' '
        if not self.spline:
            c+= '--nospline '
        #end if
        c+='--'+self.format+' '+self.h5out+' '+self.h5in
        return c
    #end def app_command
        

    def read(self,filepath):
        None
    #end def read

    def write_text(self,filepath=None):
        return self.app_command()
    #end def write_text
#end class WfconvertInput


def generate_wfconvert_input(app_name='wfconvert',h5in='MISSING.h5',h5out='wfconvert.h5',spline=False,format='eshdf',factor=None):
    wi = WfconvertInput(
        app_name = app_name,
        h5in   = h5in,
        h5out  = h5out,
        spline = spline,
        format = format,
        factor = factor
        )
    return wi
#end def generate_wfconvert_input


class WfconvertAnalyzer(SimulationAnalyzer):
    def __init__(self,arg0):
        if isinstance(arg0,Simulation):
            sim = arg0
            self.infile = sim.infile
            self.dir    = sim.locdir
            self.h5file = os.path.join(sim.locdir,sim.input.h5out)
        else:
            self.infile = arg0
        #end if
    #end def __init__

    def analyze(self):
        if False:
            import h5py
            self.log('Fixing h5 file',n=5)
            h = h5py.File(self.h5file)
            if 'electrons' in h:
                elec = h['electrons']
                nkpoints = 0
                for name,val in elec.iteritems():
                    if name.startswith('kpoint'):
                        nkpoints+=1
                    #end for
                #end if
                nkold = elec['number_of_kpoints'][0] 
                self.log('Were',nkold,'kpoints, now',nkpoints,'kpoints',n=6)
                elec['number_of_kpoints'][0] = nkpoints
            #end for        
        #end if
    #end def analyze
#end class WfconvertAnalyzer



class Wfconvert(Simulation):
    input_type             = WfconvertInput
    analyzer_type          = WfconvertAnalyzer
    generic_identifier     = 'wfconvert'
    application            = 'wfconvert'
    application_properties = set(['serial'])
    application_results    = set(['orbitals'])

    def set_app_name(self,app_name):
        self.app_name = app_name
        self.input.set_app_name(app_name)
    #end def set_app_name

    def check_result(self,result_name,sim):
        calculating_result = False
        if result_name=='orbitals':
            calculating_result = True
        else:
            calculating_result = False
            self.error('ability to check for result '+result_name+' has not been implemented')
        #end if        
        return calculating_result
    #end def check_result

    def get_result(self,result_name,sim):
        result = obj()
        if result_name=='orbitals':
            result.h5file   = os.path.join(self.locdir,self.input.h5out)
            result.outfile  = os.path.join(self.locdir,self.outfile)
        else:
            self.error('ability to get result '+result_name+' has not been implemented')
        #end if        
        return result
    #end def get_result

    def incorporate_result(self,result_name,result,sim):
        if result_name=='orbitals':
            self.input.h5in = os.path.relpath(result.h5file,self.locdir)
            self.job.app_command = self.input.app_command()
        else:
            self.error('ability to incorporate result '+result_name+' has not been implemented')
        #end if                
    #end def incorporate_result

    def check_sim_status(self):
        outfile = os.path.join(self.locdir,self.outfile)
        errfile = os.path.join(self.locdir,self.errfile)
        fobj = open(outfile,'r')
        output = fobj.read()
        fobj.close()
        fobj = open(errfile,'r')
        errors = fobj.read()
        fobj.close()
        h5file = os.path.join(self.locdir,self.input.h5out)
        file_exists = os.path.exists(h5file)
        outfin = 'Successfully read' in errors and 'numSpins' in errors
        outfin = outfin and 'Writing laplacians' in output

        success = file_exists and outfin

        self.finished = success
    #end def check_sim_status

    def get_output_files(self):
        output_files = []
        return output_files
    #end def get_output_files

    def app_command(self):
        # app_name is passed along in post_init
        return self.input.app_command()
    #end def app_command
#end class Wfconvert




def generate_wfconvert(**kwargs):
    sim_args,inp_args = Simulation.separate_inputs(kwargs)

    if not 'input' in sim_args:
        sim_args.input = generate_wfconvert_input(**inp_args)
    #end if
    wfconvert = Wfconvert(**sim_args)

    return wfconvert
#end def generate_wfconvert




# extract_fdlr_coeffs contributed by Nick Blunt
def extract_fdlr_coeffs(data_file, state, norbs, ndoub):
    '''Extract the CIS coefficients and orbital pairs for the corresponding
       single excitations, from a GAMESS output file.'''

    # Strings to search for, which mark the start of the section that we want
    # to read from.
    target_str = 'STATE #'
    state_str = '  ' + str(state) + '  '

    # String to mark the end of the CI wave function specifications.
    end_str = "...... END OF CI-MATRIX DIAGONALIZATION ......"

    # Have we found the header for the desired state?
    state_found = False
    # Are we reading lines which contain the actual data for the desired state?
    have_data = False
    # The first det has label 1, so the next one will be 2.
    next_det = 2
    # String to hold the orbital occupancy.
    orb_string = ''

    # List of all of orbitals and associated coefficients. Each element is
    # itself a list, of the form:
    # [ orb excited from, orb excited to, coefficient value ]
    coeffs = []

    f = open(data_file)

    for line in f:
        if have_data:
            values = line.split()
            next_det_str = '  ' + str(next_det) + '  '

            # If we've finished reading the current determinant's data, and
            # are about to get the next one (or are at the end of this
            # state's data section).
            if next_det_str in line or target_str in line or end_str in line:
                # This should only be true if on the first configuration.
                if orb_string != '':
                    occ_string = orb_string[0:ndoub]
                    from_orb = occ_string.find("1")

                    virt_string = orb_string[ndoub:]
                    to_orb = virt_string.find("1")

                    # If a "1" was not found, and therefore we probably have
                    # the HF det itself.
                    if from_orb == -1 or to_orb == -1:
                        coeff = float(values[1])
                        orb_string = values[2]
                        next_det += 1
                        continue
                    else:
                        # If we have a configuration to add to the list.
                        from_orb += 1
                        to_orb += ndoub + 1
                        # If the orbital excited from is not an even number of
                        # moves from the end of the occupied section, then we
                        # need to add a minus sign to the coefficient, to
                        # correct for ordering of fermionic excitations.
                        perm_test = ndoub - from_orb
                        # (Note that from_orb is zero-indexed).
                        if perm_test%2 == 0:
                            coeff *= -1.0
                        #end if
                        coeffs.append( [from_orb, to_orb, coeff] )
                        # If we're actually at the end of this state's section.
                        if target_str in line or end_str in line:
                            have_data = False
                            continue
                        #end if
                    #end if
                #end if
                # Coefficient and start of the orbital string for the
                # configuration we're now starting to read.
                coeff = float(values[1])
                orb_string = values[2]
                next_det += 1

            # If not a blank line, then add this line's orbitals to the full
            # configuration, which will in general be split over many lines.
            elif len(values) != 0:
                orb_string += values[0]
            #end if
        elif state_found:
            # If this is true then section with actual data is about to begin.
            if " --- " in line:
                have_data = True
                state_found = False
            #end if
        # If this is true then we have the start of a new state's section.
        elif target_str in line:
            # If this is true then we have the state that we want.
            if state_str in line:
                state_found = True
            # Otherwise we have a state that we don't want - stop reading in
            # and storing data for the next sections.
            else:
                state_found = False
            #end if
        #end if
    #end for

    f.close()

    coeffs.sort()

    return coeffs
#end def extract_fdlr_coeffs



class Convert4qmcInput(SimulationInput):

    input_codes = '''
        pyscf              
        qp                 
        gaussian           
        casino             
        vsvb               
        gamess             
        gamess_ascii       
        gamess_fmo         
        gamess_xml         
        '''.split()

    input_order = input_codes + '''
        prefix             
        hdf5               
        add_cusp           
        psi_tag            
        ion_tag            
        no_jastrow         
        production         
        ci                 
        read_initial_guess 
        target_state       
        natural_orbitals   
        threshold          
        zero_ci            
        add_3body_J        
        '''.split()

    input_aliases = obj(
        pyscf              = 'pyscf',
        qp                 = 'QP',
        gaussian           = 'gaussian',
        casino             = 'casino',
        vsvb               = 'VSVB',
        gamess             = 'gamess',
        gamess_ascii       = 'gamessAscii',
        gamess_fmo         = 'gamessFMO',
        gamess_xml         = 'gamesxml', # not a typo
        prefix             = 'prefix',
        hdf5               = 'hdf5',
        add_cusp           = 'addCusp',
        psi_tag            = 'psi_tag',
        ion_tag            = 'ion_tag',
        no_jastrow         = 'nojastrow',
        production         = 'production',
        ci                 = 'ci',
        read_initial_guess = 'readInitialGuess',
        target_state       = 'TargetState',
        natural_orbitals   = 'NaturalOrbitals',
        threshold          = 'threshold',
        zero_ci            = 'zeroCi',
        add_3body_J        = 'add3BodyJ',
        )

    input_types = obj(
        app_name           = str, # executable name
        pyscf              = str, # file path
        qp                 = str, # file path
        gaussian           = str, # file path
        casino             = str, # file path
        vsvb               = str, # file path
        gamess             = str, # file path
        gamess_ascii       = str, # file path
        gamess_fmo         = str, # file path
        gamess_xml         = str, # file path
        prefix             = str, # any name
        hdf5               = bool,
        add_cusp           = bool,
        psi_tag            = str, # wavefunction tag
        ion_tag            = str, # particeset tag
        no_jastrow         = bool,
        production         = bool,
        ci                 = str, # file path
        read_initial_guess = int,
        target_state       = int,
        natural_orbitals   = int,
        threshold          = float,
        zero_ci            = bool,
        add_3body_J        = bool,
        )

    input_defaults = obj(
        app_name           = 'convert4qmc',
        pyscf              = None, # input codes
        qp                 = None,
        gaussian           = None,
        casino             = None,
        vsvb               = None,
        gamess             = None, 
        gamess_ascii       = None,
        gamess_fmo         = None,
        gamess_xml         = None,
        prefix             = None, # general options
        hdf5               = False,
        add_cusp           = False,
        psi_tag            = None,
        ion_tag            = None,
        no_jastrow         = False,
        production         = False,
        ci                 = None, # gamess specific below
        read_initial_guess = None,
        target_state       = None,
        natural_orbitals   = None,
        threshold          = None,
        zero_ci            = False,
        add_3body_J        = False,# deprecated
        )

    fdlr_inputs = ['fdlr_state','norbs','double_occ','coeff_factor']

    def __init__(self,**kwargs):
        kwargs = obj(kwargs)
        # check that only allowed keyword inputs are provided
        invalid = set(kwargs.keys())-set(self.input_types.keys())-set(self.fdlr_inputs)
        if len(invalid)>0:
            self.error('invalid inputs encountered\nvalid keyword inputs are: {0}'.format(sorted(self.input_types.keys())))
        #end if

        # get fdlr inputs first, if present
        fdlr = obj()
        for k in self.fdlr_inputs:
            if k in kwargs:
                fdlr[k] = kwargs.delete(k)
            #end if
        #end for
        if len(fdlr)>0:
            fdlr.set_optional(
                coeff_factor = 0.01,
                norbs        = None,
                )
            required = ('fdlr_state','double_occ')
            for k in required:
                if k not in fdlr:
                    self.error('{0} is a required fdlr input'.format(k))
                #end if
            #end for
            self.fdlr = fdlr
        #end if

        # assign inputs
        self.set(**kwargs)

        # assign default values
        self.set_optional(**self.input_defaults)

        # check that all keyword inputs are valid
        self.check_valid()
    #end def __init__


    def check_valid(self,exit=True):
        valid = True
        # check that all inputs have valid types and assign them
        for k,v in self.iteritems():
            if v is not None and k!='fdlr' and not isinstance(v,self.input_types[k]):
                valid = False
                if exit:
                    self.error('keyword input {0} must be of type {1}\nyou provided a value of type {2}\nplease revise your input and try again'.format(k,self.input_types[k].__name__),v.__class__.__name__)
                #end if
                break
            #end if
        #end for
        return valid
    #end def check_valid


    def set_app_name(self,app_name):
        self.app_name = app_name
    #end def set_app_name


    def input_code(self):
        input_code = None
        for k in self.input_codes:
            if k in self and self[k] is not None:
                if input_code is not None:
                    input_code = None
                    break
                else:
                    input_code = self[k]
                #end if
            #end if
        #end for
        return input_code
    #end def input_code


    def has_input_code(self):
        return self.input_code() is not None
    #end def has_input_code


    def app_command(self):
        self.check_valid()
        c = self.app_name
        for k in self.input_order:
            if k in self:
                v = self[k]
                n = self.input_aliases[k]
                if isinstance(v,bool):
                    if v:
                        c += ' -{0}'.format(n)
                    #end if
                elif v is not None:
                    c += ' -{0} {1}'.format(n,str(v))
                #end if
            #end if
        #end for
        return c
    #end def app_command


    def read(self,filepath):
        None
    #end def read


    def write_text(self,filepath=None):
        return self.app_command()
    #end def write_text


    def output_files(self):
        prefix = 'sample'
        if self.prefix!=None:
            prefix = self.prefix
        #end if
        wfn_file  = prefix+'.Gaussian-G2.xml'
        ptcl_file = prefix+'.Gaussian-G2.ptcl.xml'
        return wfn_file,ptcl_file
    #end def output_files
#end class Convert4qmcInput



def generate_convert4qmc_input(**kwargs):
    return Convert4qmcInput(**kwargs)
#end def generate_convert4qmc_input



class Convert4qmcAnalyzer(SimulationAnalyzer):
    def __init__(self,arg0):
        if isinstance(arg0,Simulation):
            self.infile = arg0.infile
        else:
            self.infile = arg0
        #end if
    #end def __init__

    def analyze(self):
        None
    #end def analyze
#end class Convert4qmcAnalyzer



class Convert4qmc(Simulation):
    input_type             = Convert4qmcInput
    analyzer_type          = Convert4qmcAnalyzer
    generic_identifier     = 'convert4qmc'
    application            = 'convert4qmc'
    application_properties = set(['serial'])
    application_results    = set(['orbitals','particles','fdlr_wavefunction'])


    def set_app_name(self,app_name):
        self.app_name = app_name
        self.input.set_app_name(app_name)
    #end def set_app_name


    def propagate_identifier(self):
        None
        #self.input.prefix = self.identifier
    #end def propagate_identifier


    def get_prefix(self):
        input = self.input
        prefix = 'sample'
        if input.prefix is not None:
            prefix = input.prefix
        #end if
        return prefix
    #end def get_prefix


    def list_output_files(self):
        # try to support both pre and post v3.3.0 convert4qmc
        prefix = self.get_prefix()
        wfn_file  = prefix+'.Gaussian-G2.xml'
        ptcl_file = prefix+'.Gaussian-G2.ptcl.xml'
        if not os.path.exists(os.path.join(self.locdir,ptcl_file)):
            wfn_file  = prefix+'.wfnoj.xml'
            ptcl_file = prefix+'.structure.xml'
        #end if
        return wfn_file,ptcl_file
    #end def list_output_files


    def check_result(self,result_name,sim):
        calculating_result = False
        if result_name=='orbitals':
            calculating_result = True
        elif result_name=='particles':
            calculating_result = True
        elif result_name=='fdlr_wavefunction':
            calculating_result = 'fdlr' in self.input
        else:
            calculating_result = False
            self.error('ability to check for result '+result_name+' has not been implemented')
        #end if        
        return calculating_result
    #end def check_result


    def get_result(self,result_name,sim):
        result = obj()
        input = self.input
        wfn_file,ptcl_file = self.list_output_files()
        if result_name=='orbitals':
            result.location = os.path.join(self.locdir,wfn_file)
            if self.input.hdf5==True:
                orbfile = self.get_prefix()+'.orbs.h5'
                result.orbfile = os.path.join(self.locdir,orbfile)
            #end if
        elif result_name=='particles':
            result.location = os.path.join(self.locdir,ptcl_file)
        elif result_name=='fdlr_wavefunction':
            wfn_d = self.input.prefix+'.wfn_d.xml'
            wfn_x = self.input.prefix+'.wfn_x.xml'
            result.wfn_d = os.path.join(self.locdir,wfn_d) 
            result.wfn_x = os.path.join(self.locdir,wfn_x) 
        else:
            self.error('ability to get result '+result_name+' has not been implemented')
        #end if        
        return result
    #end def get_result


    def incorporate_result(self,result_name,result,sim):
        implemented = True
        if isinstance(sim,Gamess):
            if result_name=='orbitals':
                input = self.input
                orbpath = os.path.relpath(result.location,self.locdir)
                if result.scftyp=='mcscf':
                    input.gamess_ascii = orbpath
                    input.ci           = orbpath
                elif result.scftyp=='none': # cisd, etc
                    input.gamess_ascii = orbpath
                    input.ci           = orbpath
                    if result.mos>0:
                        input.read_initial_guess = result.mos
                    elif result.norbitals>0:
                        input.read_initial_guess = result.norbitals
                    #end if
                else:
                    input.gamess_ascii = orbpath
                #end if
                self.job.app_command = input.app_command()
            else:
                implemented = False
            #end if
        else:
            implemented = False
        #end if
        if not implemented:
            self.error('ability to incorporate result {0} from {1} has not been implemented'.format(result_name,sim.__class__.__name__))
        #end if
    #end def incorporate_result


    def check_sim_status(self):
        output = open(os.path.join(self.locdir,self.outfile),'r').read()
        #errors = open(os.path.join(self.locdir,self.errfile),'r').read()

        success = 'QMCGaussianParserBase::dump' in output
        for filename in self.list_output_files():
            success &= os.path.exists(os.path.join(self.locdir,filename))
        #end for

        self.failed = not success
        self.finished = self.job.finished
    #end def check_sim_status


    def get_output_files(self):
        output_files = []
        return output_files
    #end def get_output_files


    def app_command(self):
        return self.input.app_command()
    #end def app_command


    def post_analyze(self,analyzer):
        input = self.input
        # prepare fdlr wavefunction files
        if 'fdlr' in input:
            if self.input.hdf5:
                self.error('fdlr is currently incompatible with orbitals stored in hdf5 format')
            #end if

            # get fdlr inputs
            fdlr = input.fdlr
            norbs = fdlr.norbs
            stored_input = self.load_input_image() # input w/ orbital count info
            if norbs is None:
                norbs = stored_input.read_initial_guess
            #end if

            # locate gamess output
            gms_out = os.path.abspath(os.path.join(self.locdir,stored_input.gamess_ascii))
            if not os.path.exists(gms_out):
                self.warn('GAMESS output file does not exist\nfilepath: {0}\ncannot proceed with FDLR wavefunction'.format(gms_out))
                self.failed = True
                return
            #end if

            # extract state coefficients
            failed = False
            try:
                coeffs = extract_fdlr_coeffs(
                    data_file = gms_out,
                    state     = fdlr.fdlr_state,
                    norbs     = norbs,
                    ndoub     = fdlr.double_occ,
                    )
            except:
                failed = True
            #end try
            if failed or len(coeffs)==0:
                self.warn('FDLR state coefficient extraction failed\ncannot proceed with FDLR wavefunction')
                self.failed = True
                return
            #end if
            coeffs  = fdlr.coeff_factor*array(coeffs,dtype=float)[:,2]
            zcoeffs = zeros(coeffs.shape)
            
            # open wavefunction file made by convert4qmc
            wfn_file,ptcl_file = self.list_output_files()
            temp_file = os.path.join(self.locdir,wfn_file)
            if not os.path.exists(temp_file):
                self.warn('QMCPACK input file does not exist\nfilepath: {0}\ncannot proceed with FDLR wavefunction'.format(temp_file))
                self.failed = True
                return
            #end if
            tf = TextFile(temp_file)

            # get determinantset block
            tf.seek('<wavefunction',0)
            tf.seek('\n',1)
            i1 = tf.tell()+1
            tf.seek('</basisset>',1)
            tf.seek('\n',1)
            i2 = tf.tell()+1
            dset = tf[i1:i2]
            
            # get up det contents
            tf.seek('<determinant',1)
            tf.seek('\n',1)
            i1 = tf.tell()+1
            tf.seek('</coefficient>',1)
            tf.seek('\n',1)
            i2 = tf.tell()+1
            updet = tf[i1:i2]
            
            # get down det contents
            tf.seek('<determinant',1)
            tf.seek('\n',1)
            i1 = tf.tell()+1
            tf.seek('</coefficient>',1)
            tf.seek('\n',1)
            i2 = tf.tell()+1
            dndet = tf[i1:i2]

            # create wfn_d and wfn_x files
            for wfn,coef in [('wfn_d',coeffs),('wfn_x',zcoeffs)]:
                scoef = ''
                for cv in coef:
                    scoef += '           {0: 16.8e}\n'.format(cv)
                #end for
                c = '<?xml version="1.0"?>\n'
                c += '<{0}>\n'.format(wfn)
                c += dset
                c += '     <slaterdeterminant optimize="yes">\n'
                c += '       <determinant id="det_up" sposet="spo-up">\n'
                c += '         <opt_vars size="{0}">\n'.format(len(coef))
                c += scoef
                c += '         </opt_vars>\n'
                c += '       </determinant>\n'
                c += '       <determinant id="det_down" sposet="spo-dn">\n'
                c += '         <opt_vars size="{0}">\n'.format(len(coef))
                c += scoef
                c += '         </opt_vars>\n'
                c += '       </determinant>\n'
                c += '     </slaterdeterminant>\n'
                c += '     <sposet basisset="LCAOBSet" name="spo-up" size="{0}" optimize="yes">\n'.format(norbs)
                c += updet
                c += '     </sposet>\n'
                c += '     <sposet basisset="LCAOBSet" name="spo-dn" size="{0}" optimize="yes">\n'.format(norbs)
                c += dndet
                c += '     </sposet>\n'
                c += '  </determinantset>\n'
                c += '</{0}>\n'.format(wfn)
                filename = '{0}.{1}.xml'.format(self.input.prefix,wfn)
                filepath = os.path.join(self.locdir,filename)
                fobj = open(filepath,'w')
                fobj.write(c)
                fobj.close()
            #end for

            tf.close()
        #end if
    #end def post_analyze
#end class Convert4qmc



def generate_convert4qmc(**kwargs):
    sim_args,inp_args = Simulation.separate_inputs(kwargs)
    if 'identifier' in sim_args and not 'prefix' in inp_args:
        inp_args.prefix = sim_args.identifier
    #end if

    if not 'input' in sim_args:
        sim_args.input = generate_convert4qmc_input(**inp_args)
    #end if
    convert4qmc = Convert4qmc(**sim_args)

    return convert4qmc
#end def generate_convert4qmc
