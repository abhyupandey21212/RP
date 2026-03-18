# IntrinsicBeamReduction


IntrinsicBeamNL.py
The file IntrinsicBeamNL defines the class BeamNL which is an FEM implementation of the geometrically exact beam equations. 
3-node elements with quadratic shape functions are implemented. Beam is assumed to be isotropic, although it should be simple to adapt the code to work for beams with changing physical properties.
It must be initated with the number of nodes and some physical properties. 
It contains many helper (private) methods as well as the following public methods:
	static_solver - solves the static problem for a given external force
	post - does post processing to go from intrinsic to displacement and finite-rotation
	dynamic_solver - incomplete
	coupled_solver - solves a coupled aeroelastic problem using an AeroStrip object.

It also has the BeamPOD class, which implements POD-reduction of the beam.
It is initiated with a BeamNL object, whose helper function and physical properties it uses. On top of that it has:
	POD_offline - does the offline computations needed to use POD, precompuptes all reduced matrices
	static_solver_POD - static solver with POD reduction implemented
	dynamic_solver_POD - incomplete
	coupled_solver_POD - incomplete
Both classes by default use the Kronecker formulation. The Jacobian is also implemented using the Kronecker formulation and is used by default by all solver, but can be disables by passes the argument anal_jac=False to the solver. The standard formulation can be used by passing the argument legacy=True

aeroelastics.py
Defines the class AeroStrip, a simple linear strip aerodynamic model for the coupled_solver.

IntrinsicBeamSampler.py
Defines a sampler function that takes a BeamNL object and solves many samples to be used for POD. It can be given the template force vector, which it multiples by a random float chosen uniformly between [-P_max, P_max]. Uses multithreading for efficiency. 

Contains a Tester class, which can be given a BeamNL and BeamPOD object and tests the performance of the ROM relative to the FOM.
Contains a mesh convergence testing function.

LBeamFEM.py
Contains an FEM model of a linear Euler beam as well as an analytical solution for tip foces in a cantilever configuration. 
Used for validating the models in IntrinsicBeamNL.

dimentions.py
Contains all the physical parameters, taken from the Pazy wing. See Pazy_wing_data.png


