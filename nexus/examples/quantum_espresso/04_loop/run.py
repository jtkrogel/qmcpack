#! /usr/bin/env python3
"""
Enhanced loop workflow for Quantum ESPRESSO.

This script demonstrates the loop support added to enhanced simulations.  A
loop-enabled SCF step repeatedly resubmits itself until either a convergence
condition is met or the maximum number of iterations is reached.

"""

from nexus import settings, job, run_project
from nexus import generate_physical_system
from nexus import generate_pwscf

from enhanced_simulation import make_enhanced

# Computer configuration
computer = 'ws8'


# Loop behavior notes:
# - Loops reuse the same directory and identifier (unlike error handlers which create
#   new paths like "attempt_1", "attempt_2", etc.)
# - Simulation ID (simid) remains constant across all loop iterations
# - Each iteration gets a NEW scheduler job ID (process_id) when resubmitted
#   (process_id is updated via update_process_id() when the new job is submitted,
#   replacing the previous scheduler ID)
# - iteration_count tracks current loop iteration (0 = first run, 1 = second, etc.)
# - Status output shows [iter=N] for simulations with iteration_count > 0
# - Access iteration_count programmatically: sim.iteration_count
# - Each iteration overwrites files in the same directory (no attempt subdirectories)
# - Loop continues until loop_condition returns False OR loop_max_iterations is reached

LOOP_TARGET = 3

settings(
    pseudo_dir = '../pseudopotentials',
    results    = '',
    sleep      = 1,
    machine    = computer,
    )


system = generate_physical_system(
    units    = 'A',
    axes     = '''1.785   1.785   0.000
                  0.000   1.785   1.785
                  1.785   0.000   1.785''',
    elem_pos = '''
               C  0.0000  0.0000  0.0000
               C  0.8925  0.8925  0.8925
               ''',
    C        = 4,
    )


# Seed calculation (plain DAG simulation)

scf_inputs = dict(
    job          = job(cores=4,app='pw.x'),
    input_type   = 'generic',
    calculation  = 'scf',
    input_dft    = 'lda',
    ecutwfc      = 200,
    conv_thr     = 1e-8,
    system       = system,
    pseudos      = ['C.BFD.upf'],
    kgrid        = (2,2,1),
    kshift       = (0,0,0),
    )

scf_seed = generate_pwscf(
    identifier   = 'scf_seed',
    path         = 'diamond/scf_seed',
    **scf_inputs
    )


def continue_loop(sim):
    """
    Loop condition: keep iterating until the target count is reached.
    The sim.iteration_count is incremented automatically after each pass.
    """
    return sim.iteration_count < LOOP_TARGET


def modify_loop_input(sim):
    """
    Loop modifier: modify simulation input at each iteration.
    
    This function is called before each loop iteration (after iteration_count
    is incremented but before the simulation is resubmitted). You can modify
    any simulation attributes here, such as:
    - Input parameters (ecutwfc, conv_thr, etc.)
    - System geometry
    - K-point grid
    - Any other simulation attributes
    
    Example: Gradually increase convergence threshold at each iteration
    """
    # Example: Modify convergence threshold based on iteration
    # Start with loose threshold, tighten as iterations progress
    base_threshold = 1e-4
    iteration_factor = 10 ** (-sim.iteration_count)
    sim.input.electrons.conv_thr = base_threshold * iteration_factor
    
    # Example: Store iteration-specific values in loop_variables
    if not hasattr(sim.loop_variables, 'modified_conv_thr'):
        sim.loop_variables['modified_conv_thr'] = [sim.input.electrons.conv_thr]
    else:
        sim.loop_variables['modified_conv_thr'].append(sim.input.electrons.conv_thr)



loop_base = generate_pwscf(
    identifier   = 'scf_loop',
    path         = 'diamond/scf_loop',
    dependencies = (scf_seed, 'charge_density'),
    **scf_inputs
    )

scf_loop = make_enhanced(
    loop_base,
    loop_enabled        = True,
    loop_condition      = continue_loop,
    loop_max_iterations = LOOP_TARGET,
    loop_modifier       = modify_loop_input,  # Modify sim at each iteration
    )

run_project()

