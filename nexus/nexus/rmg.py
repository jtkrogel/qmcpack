##################################################################
##  (c) Copyright 2020-  by Jaron T. Krogel                     ##
##################################################################


import glob
import os
import re

from .developer import obj
from .simulation import Simulation
from .pseudoset import PseudoSet
from .rmg_input import RmgInput, generate_rmg_input
from .rmg_analyzer import RmgAnalyzer



class Rmg(Simulation):
    input_type             = RmgInput
    analyzer_type          = RmgAnalyzer
    generic_identifier     = 'rmg'
    application            = 'rmg-cpu'
    application_properties = frozenset({'serial','mpi'})
    application_results    = frozenset({'orbitals','mfr_potential','mfr_core_action','mfr_meta_fields'})


    def check_result(self,result_name,sim):
        mfr = getattr(self.input,'write_mfr_potential',False)
        mfr_core = getattr(self.input,'write_mfr_core_action',False)
        mfr_meta = getattr(self.input,'write_mfr_meta_fields',False)
        if result_name=='orbitals':
            return mfr or mfr_core or mfr_meta or getattr(self.input,'write_qmcpack_restart',False)
        elif result_name=='mfr_potential':
            return mfr
        elif result_name=='mfr_core_action':
            return mfr_core
        elif result_name=='mfr_meta_fields':
            return mfr_meta
        #end if
        return False
    #end def check_result


    def get_result(self,result_name,sim):
        if not self.check_result(result_name,sim):
            raise ValueError('RMG is not calculating result '+result_name)
        #end if
        prefix = getattr(self.input,'output_wave_function_file','Waves/wave.out')
        prefix = os.path.abspath(os.path.join(self.locdir,prefix))
        if result_name=='orbitals':
            return obj(h5file=prefix+'.h5')
        elif result_name=='mfr_potential':
            return obj(h5file=prefix+'.mfr.h5')
        elif result_name=='mfr_core_action':
            return obj(h5file=prefix+'.mfr_core.h5')
        elif result_name=='mfr_meta_fields':
            return obj(h5file=prefix+'.mfr_meta.h5')
        #end if
    #end def get_result


    def incorporate_result(self,result_name,result,sim):
        msg = 'ability to incorporate result '+result_name+' has not been implemented'
        raise NotImplementedError(msg)
    #end def incorporate_result


    def app_command(self):
        return self.app_name+' '+self.infile
    #end def app_command


    def check_sim_status(self):
        # Each invocation increments the numbered log. A successful older log
        # must not release dependencies after a failed retry.
        logs = sorted(glob.glob(os.path.join(self.locdir,self.infile+'.[0-9][0-9].log')))
        output = ''
        if logs:
            with open(logs[-1]) as f:
                output = f.read()
            #end with
        #end if
        completed = 'TIMING INFORMATION FOR THREADS' in output
        converged = 'potential convergence has been achieved' in output.lower()
        if getattr(self.input,'calculation_mode','Quench Electrons')=='Quench Electrons':
            completed = completed and converged
        #end if
        for result_name in self.application_results:
            if self.check_result(result_name,None):
                path = self.get_result(result_name,None).h5file
                completed = completed and os.path.isfile(path) and os.path.getsize(path)>0
            #end if
        #end for
        export_markers = (
            ('mfr_potential', 'Wrote converged MFR potential to '),
            ('mfr_core_action', 'Wrote converged MFR core actions to '),
            ('mfr_meta_fields', 'Wrote converged MFR meta fields to '),
            )
        for result_name,marker in export_markers:
            if not self.check_result(result_name,None):
                continue
            #end if
            expected_path = os.path.abspath(self.get_result(result_name,None).h5file)
            marker_present = False
            for line in output.splitlines():
                line = line.strip()
                # Native rmg_printf can prepend its progress/source tag.
                # Bind every writer marker to this result, so a message for
                # another prefix cannot release a stale expected file.
                match = re.match(r'^(?:\[[^\]]+\]\s+[^:]+:\s+)?'+re.escape(marker)+r'(.+)$',line)
                if match:
                    reported_path = match.group(1).strip()
                    reported_path = os.path.abspath(os.path.join(self.locdir,reported_path))
                    if reported_path == expected_path:
                        marker_present = True
                        break
                    #end if
                #end if
            #end for
            completed = completed and marker_present
        #end for
        self.succeeded = bool(self.job.finished and completed)
        self.failed    = bool(self.job.finished and not completed)
        self.finished  = bool(self.job.finished)
    #end def check_sim_status


    def get_output_files(self): # returns list of output files to save
        files = []
        for result_name in sorted(self.application_results):
            if self.check_result(result_name,None):
                path = self.get_result(result_name,None).h5file
                files.append(os.path.relpath(path,self.locdir))
            #end if
        #end for
        return files
    #end def get_output_files
#end class Rmg




def generate_rmg(**kwargs):
    pseudos = kwargs.get('pseudos',None)
    if pseudos is not None:
        system = kwargs.get('system',None)
        pseudos = PseudoSet.get_pseudos(
            pseudos = pseudos,
            system = system,
            code = 'rmg',
            )
        kwargs['pseudos'] = pseudos
        kwargs['files'] = list(kwargs.get('files',[])) + list(pseudos.values())
    #end if

    sim_args,inp_args = Rmg.separate_inputs(kwargs)

    if 'input' not in sim_args:
        sim_args.input = generate_rmg_input(**inp_args)
    #end if
    rmg = Rmg(**sim_args)

    return rmg
#end def generate_rmg
