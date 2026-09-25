##################################################################
##  (c) Copyright 2020-  by Jaron T. Krogel                     ##
##################################################################


from __future__ import annotations

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
    application_results    = frozenset({''})


    def check_result(self, result_name: str, sim: Simulation) -> bool:
        calculating_result = False
        return calculating_result
    #end def check_result

    #mth
    def get_result(self, result_name: str, sim: Simulation):
        result = None
        msg = 'Ability to get result '+result_name+' has not been implemented.'
        raise NotImplementedError(msg)
        return result
    #end def get_result

    #mth
    def incorporate_result(
        self,
        result_name : str,
        result,  #th
        sim         : Simulation,
        ) -> None:
        msg = 'ability to incorporate result '+result_name+' has not been implemented'
        raise NotImplementedError(msg)
    #end def incorporate_result


    def app_command(self) -> str:
        return self.app_name+' '+self.infile
    #end def app_command


    def check_sim_status(self) -> None:
        # assume all is well
        self.succeeded = True
        self.failed    = False
        self.finished  = self.job.finished
    #end def check_sim_status


    def get_output_files(self) -> list: # returns list of output files to save
        return []
    #end def get_output_files
#end class Rmg



#mth
def generate_rmg(**kwargs) -> Rmg:
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
