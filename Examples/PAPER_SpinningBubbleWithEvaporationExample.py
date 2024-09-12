import os
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import sympy as sp
from PhysicalModels.ThermoPhysicalModels.EOS_factory import StiffenedGasEOS
from PhysicalModels.ThermoPhysicalModels.TransportProperties_factory import ConstantFluidTransportProperty
from PhysicalModels.CompressibleNavierStokesModels import CompressibleNavierStokesModels
from ManufSolution.ManufSolShape.PolarPolynomialManufSol import PolarPolynomialManufSol
from ManufSolution.ManufSolShape.UserDefinedManufSol import UserDefinedManufSol
from ManufSolution.ManufSolContainer.CompressibleFlowManufSolContainer import CompressibleFlowManufSolContainer
from ManufSolution.ManufSolContainer.GeneralManufSolContainer import GeneralManufSolContainerMultiplePhases
from Common.MMSTags import CompressibleFlowVarSetTags, FluidSolutionTags, FluidSolutionGradientTags, ProjectionType
from BoundaryConditions.BoundaryGeometry import BoundaryGeometryFromEquation, BoundaryGeometryFromPoints
from BoundaryConditions.GeneralBoundaryConditions import GeneralBoundaryConditions
from Common.MMSTags import CoordinatesSystemType, OutputFileType
from ProblemSolving.GeneralProblemHandler import GeneralProblemHandler, QuantityInfoForPlot
from Outputs.PlotOverArea import *

current_directory = os.getcwd()
output_folder_name = "Examples/PAPER_SpinningBubbleWithEvaporationOutput/"
output_absolute_path = os.path.join(current_directory,output_folder_name)

#* -------------------------------------------------------------------------- *#
#* --- 0/ DEFINITION OF PROBLEM PARAMETERS ---
#* -------------------------------------------------------------------------- *#
pressureHeatFluxDecoupled = False

# Geometry definition
domain_dim = 2
L = [0.5,1.5]
H = [0.5,1.5]
center = [1.0, 1.0]
radius = 0.25625#0.25

# Reference solution values 

# m_dot = -0.3 # using m_dot != 0 leads to a non-solvable system...
# deltaTangentStress = 0.2
# deltaTangentVel = -0.5
# deltaTemp = 10.0
# heatSource = 100.0

m_dot = -0.001 # using m_dot != 0 leads to a non-solvable system...
deltaTangentStress = 0.2
deltaTangentVel = -50.0
deltaTemp = 100.0
heatSource = 1000.0


p_far = 20000
# Interfacial jumps
jump_p = 0.01*p_far #sigma/radius #=> if jump_p != 0, need to take it into account in energy jump and this seems to expensive for symbolic solver 

# Fixed physical parameters

delta_p_far_int = 0.1*p_far
p_int_2 = p_far+delta_p_far_int # pressure in the gas at the interface
p_int_1 = p_int_2 + jump_p # pressure in the liquid at the interface

# Liquid physical properties
mu_1 = 4.45E-3
lambda_1 = 35.242
Cv_1 = 977.704
p_inf_1 = 7.16E9
T_melting_liquid = 1941
rho_cst_liquid = 4140
EOS_1_interm = StiffenedGasEOS(0.0, Cv_1, p_inf_1, 2)
gamma_0 = EOS_1_interm.getGammaFromPTRho(p_int_1, T_melting_liquid, rho_cst_liquid)
print("==>> gamma_liquid = %2.16f"%gamma_0)
EOS_1 = StiffenedGasEOS(gamma_0, Cv_1, p_inf_1, 2)

# Gas physical properties
mu_2 = 0.00006982
lambda_2 = 0.126825
Cv_2 = 991.20
Cp_2 = 1278.25
gamma_2 = Cp_2/Cv_2
print("==>> gamma_gas = %2.16f"%gamma_2)
p_inf_2 = 0.0
EOS_2 = StiffenedGasEOS(gamma_2, Cv_2, p_inf_2, 2) 

phys_param = {0: [gamma_0, Cv_1, p_inf_1, mu_1, lambda_1], 1: [gamma_2, Cv_2, p_inf_2, mu_2, lambda_2]}

# Physical conditions at the core of the liquid
rho_ref = 0.999*rho_cst_liquid

p_ref = 0.99*p_int_1
T_ref = EOS_1.T_rhoP(rho_ref, p_ref)
u_r_ref = 0.0
u_theta_ref = 0.0

# Physical conditions in the gas far field
T_far = 2500
rho_far = EOS_2.rho_pT(p_far, T_far)
u_r_far = 10.0
u_theta_far = 10.0


# Interfacial physical conditions
rho_int_1 = rho_cst_liquid
T_int = T_melting_liquid
u_r_int = 1.0
u_theta_int = 1.0
rho_int_2 = EOS_2.rho_pT(p_int_2, T_int)
q_dot = heatSource

U_1_prim = [p_int_1, u_r_int, u_theta_int, T_int]
# U_1_cons = EOS_1.primitivePVTToConservative(U_1_prim)
# c_1 = EOS_1.computeSoundSpeed(U_1_cons)
# Mach_1 = u_r_int/c_1
# print("------ Mach in liquid @ interface = %2.6f ------"%Mach_1)

# print("========= LIQUID @ interface ==========")
# print("------ Rho = %2.6f ------"%rho_int_1)
# print("------ P = %2.6f ------"%p_int_1)
# print("------ T = %2.6f ------"%T_int)

# print("========= LIQUID in core ==========")
# print("------ Rho = %2.6f ------"%rho_ref)
# print("------ P = %2.6f ------"%p_ref)
# print("------ T = %2.6f ------"%T_ref)

# print("========= Gas @ interface ==========")
# print("------ Rho = %2.6f ------"%rho_int_2)
# print("------ P = %2.6f ------"%p_int_2)
# print("------ T = %2.6f ------"%T_int)

# print("========= Gas in far field ==========")
# print("------ Rho = %2.6f ------"%rho_far)
# print("------ P = %2.6f ------"%p_far)
# print("------ T = %2.6f ------"%T_far)

T_refs = [T_ref, T_int, T_far]
p_refs = [p_ref, p_int_1, p_far]
u_r_refs = [u_r_ref, u_r_int, u_r_far]
u_theta_refs = [u_theta_ref, u_theta_int, u_theta_far]

p_ref = p_refs[0]
p_int = p_refs[1]
p_far = p_refs[2]
T_ref = T_refs[0]
T_int = T_refs[1]
T_far = T_refs[2]
u_ref = u_r_refs[0]
u_int = u_r_refs[1]
u_far = u_r_refs[2]
v_ref = u_theta_refs[0]
v_int = u_theta_refs[1]
v_far = u_theta_refs[2]
R_far = 3.0*radius

#* -------------------------------------------------------------------------- *#
#* --- 1/ MANUFACTURED SOLUTION ASSEMBLY ---
#* -------------------------------------------------------------------------- *#

sym_variables = sp.symbols('r, theta')

known_ind = [3, 4]

# Temperatures
temp_0 = PolarPolynomialManufSol([], sym_variables, [T_ref, 0, 0, (T_int-T_ref)/radius**3, 0], [0, 1, 2, 3, 4], 4)

sym_param_T1 = list(sp.symbols('bT_2, bT_3, bT_4'))
temp_1 = PolarPolynomialManufSol(sym_param_T1, sym_variables, [0, 0], known_ind, 4)

# Radial velocities
u_r_0 = PolarPolynomialManufSol([], sym_variables, [u_ref, 0, 0, (u_int-u_ref)/radius**3, 0], [0, 1, 2, 3, 4], 4)

sym_param_u_r_1 = list(sp.symbols('bU_2, bU_3, bU_4'))
u_r_1 = PolarPolynomialManufSol(sym_param_u_r_1, sym_variables, [0, 0], known_ind, 4)

# Tangential velocities
u_theta_0 = PolarPolynomialManufSol([], sym_variables, [v_ref, 0, 0, (v_int-v_ref)/radius**3, 0], [0, 1, 2, 3, 4], 4)

sym_param_u_theta_1 = sp.symbols('bV_2, bV_3, bV_4')
u_theta_1 = PolarPolynomialManufSol(sym_param_u_theta_1, sym_variables, [0, 0], known_ind, 4)

# Pressures
p_0_user = p_ref + (p_int - p_ref)*sp.Pow(sym_variables[0]/radius,2)
p_0 = UserDefinedManufSol([], sym_variables, p_0_user)
p_int_otherSide = p_int
sym_param_p_1 = []
p_1_b = (p_far-p_int)*radius**1/(R_far**1-radius**1)
p_1_a = p_int_otherSide - p_1_b
sym_param_p_1 = [sp.symbols('bP_1')] 
p_1_user = p_1_a + p_1_b*sp.Pow(sym_variables[0]/radius,2) + sym_param_p_1[0]
p_1 = UserDefinedManufSol(sym_param_p_1, sym_variables, p_1_user)


# Assembly into dict
dim = 2
manuf_sol_list_0 = [[p_0], [u_r_0, u_theta_0], [temp_0]]
manuf_sol_list_1 = [[p_1], [u_r_1, u_theta_1], [temp_1]]
manuf_sol_cont_0 = CompressibleFlowManufSolContainer(manuf_sol_list_0, domain_dim, sym_variables, CompressibleFlowVarSetTags.PRIMITIVE_PVT, [])
manuf_sol_cont_1 = CompressibleFlowManufSolContainer(manuf_sol_list_1, domain_dim, sym_variables, CompressibleFlowVarSetTags.PRIMITIVE_PVT, [])
dict_symb_sols = GeneralManufSolContainerMultiplePhases([manuf_sol_cont_0, manuf_sol_cont_1], [0, 1])

# dict_symb_sols_object = CompressibleFlowManufSolContainer([manuf_sol_list_0, manuf_sol_list_1], CompressibleFlowVarSetTags.PRIMITIVE_PVT)
# dict_symb_sols = dict_symb_sols_object.assembleManufSolListsIntoDic()
print(dict_symb_sols.getDicManufSolPerPhaseAndTag())

#* -------------------------------------------------------------------------- *#
#* --- 2/ DEFINITION OF THE GOVERNING EQUATIONS ---
#* -------------------------------------------------------------------------- *#

# Equation of states
Cv_0 = 977.704
p_inf_0 = 7.16E9
EOS_phase_0 = StiffenedGasEOS(gamma_0, Cv_0, p_inf_0, domain_dim)

Cv_1 = 991.20
Cp_1 = 1278.25
gamma_1 = Cp_2/Cv_2
p_inf_1 = 0.0
EOS_phase_1 = StiffenedGasEOS(gamma_1, Cv_1, p_inf_1, domain_dim) 

# Transport properties
mu_0 = 4.45E-3
lambda_0 = 35.242
transp_prop_0 = ConstantFluidTransportProperty(mu_0, lambda_0)

mu_1 = 0.00006982
lambda_1 = 0.126825
transp_prop_1 = ConstantFluidTransportProperty(mu_1, lambda_1)

# Equations models
name_model_0 = "phase0"
eqs_0 = CompressibleNavierStokesModels(name_model_0, manuf_sol_cont_0, domain_dim, EOS_phase_0, transp_prop_0, CoordinatesSystemType.CYLINDRICAL)

name_model_1 = "phase1"
eqs_1 = CompressibleNavierStokesModels(name_model_1, manuf_sol_cont_1, domain_dim, EOS_phase_1, transp_prop_1, CoordinatesSystemType.CYLINDRICAL)

# print(eqs_0.getSolTag(FluidSolutionTags.VELOCITY_VEC))
# print(eqs_0.getDerivSolTag(FluidSolutionTags.VELOCITY_X, [[sym_variables[0]]]))
# print(eqs_0.getDerivSolTag(FluidSolutionTags.VELOCITY_VEC, [[sym_variables[0]],[sym_variables[1]]]))
# print(eqs_0.getGradSolTag(FluidSolutionTags.VELOCITY_VEC, sym_variables))
# print(eqs_1.getGradSolTag(FluidSolutionTags.VELOCITY_VEC, sym_variables))
# print(eqs_0.getGradSolTag(FluidSolutionTags.VELOCITY_X, sym_variables))
# print(eqs_0.getDivergenceSolTag(FluidSolutionTags.VELOCITY_VEC, sym_variables))
# print(eqs_0.getGradVarTag(FluidSolutionGradientTags.SHEARSTRESS))
# print(eqs_0.getGradVarTag(FluidSolutionGradientTags.HEATFLUX))
# print(eqs_0.diffusiveFlux())
# print(eqs_1.diffusiveFlux())

#* -------------------------------------------------------------------------- *#
#* --- 3/ DEFINITION OF THE BOUNDARY AND JUMPS CONDITIONS (OPTIONAL!) ---
#* -------------------------------------------------------------------------- *#

# Definition of boundaries geometry
iso0_geo = BoundaryGeometryFromEquation(sym_variables, [1.0, 0.0, radius], "iso0")
far_field_geo = BoundaryGeometryFromEquation(sym_variables, [1.0, 0.0, R_far], "farField")
coord_to_subsituted = sym_variables[0]

# === Jumps on interface ===
iso0_jump = GeneralBoundaryConditions([eqs_0, eqs_1], iso0_geo)

# -> Dirichlet jumps on solutions

# - Normal velocity
jump_v_n = 0.0
rho_0 = eqs_0.getSolTag(FluidSolutionTags.DENSITY)
rho_1 = eqs_1.getSolTag(FluidSolutionTags.DENSITY)
if m_dot != 0.0:
    jump_inv_rho = ((1.0/rho_0) - (1.0/rho_1))
    jump_v_n = m_dot * jump_inv_rho
iso0_jump.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_VEC, [jump_v_n], coord_to_subsituted, ProjectionType.NORMAL)

# - Tangential velocity
iso0_jump.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_VEC, [deltaTangentVel], coord_to_subsituted, ProjectionType.TANGENT)

# - Temperature
iso0_jump.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.TEMPERATURE, [deltaTemp], coord_to_subsituted)

# - Pressure
jump_pressure = jump_p - m_dot * jump_v_n
iso0_jump.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.PRESSURE, [jump_pressure], coord_to_subsituted)

# -> Neumann jumps on fluxes

# - Normal shear stress
p_0 = eqs_0.getSolTag(FluidSolutionTags.PRESSURE)
p_1 = eqs_1.getSolTag(FluidSolutionTags.PRESSURE) 
# -[tau_nn] = 0.0 -> [p] = sigma*kappa - m_dot * [u_n]
jump_tau_nn = 0.0
iso0_jump.addNeumannBCAndSubsituteBoundaryEq(FluidSolutionGradientTags.SHEARSTRESS, [-jump_tau_nn], coord_to_subsituted, ProjectionType.NORMALNORMAL)

# - Tangential shear stress
jump_tau_nt = -m_dot * deltaTangentVel + deltaTangentStress
iso0_jump.addNeumannBCAndSubsituteBoundaryEq(FluidSolutionGradientTags.SHEARSTRESS, [-jump_tau_nt], coord_to_subsituted, ProjectionType.NORMALTANGENT)

# - Heat flux minus viscous dissipation
E_0 = eqs_0.getSolTag(FluidSolutionTags.TOTALENERGY)
E_1 = eqs_1.getSolTag(FluidSolutionTags.TOTALENERGY)
vel_0 = eqs_0.getSolTag(FluidSolutionTags.VELOCITY_VEC)
vel_1 = eqs_1.getSolTag(FluidSolutionTags.VELOCITY_VEC)
n_gamma = iso0_jump.getBoundaryGeometry().getNormalToBoundary()
t_gamma = iso0_jump.getBoundaryGeometry().getTangentsToBoundary()
vel_0_n_gamma = sum([a*b for a,b in zip(vel_0,n_gamma)])
vel_1_n_gamma = sum([a*b for a,b in zip(vel_1,n_gamma)])
vel_0_t_gamma = sum([a*b for a,b in zip(vel_0,t_gamma[0])])
vel_1_t_gamma = sum([a*b for a,b in zip(vel_1,t_gamma[0])])
u_n_ave = 0.5 * (vel_0_n_gamma+vel_1_n_gamma)
overRho_ave = 0.5 * (1.0/rho_0 + 1.0/rho_1)
u_gamma_n = u_n_ave - m_dot * overRho_ave 
u_gamma_t = 0.5 * (vel_0_t_gamma+vel_1_t_gamma)   
jump_q_minus_tau = -m_dot * (E_0-E_1) - (p_0*vel_0_n_gamma-p_1*vel_1_n_gamma) + jump_p * u_gamma_n + u_gamma_t * deltaTangentStress + heatSource
iso0_jump.addNeumannBCAndSubsituteBoundaryEq(FluidSolutionGradientTags.HEATFLUX_MINUS_VISCOUSDISSIPATION, [jump_q_minus_tau], coord_to_subsituted, ProjectionType.NORMAL)

# === Boundary conditions in the far field ===
far_field_bc = GeneralBoundaryConditions([eqs_1], far_field_geo)

far_field_bc.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.TEMPERATURE, [T_far], coord_to_subsituted)
far_field_bc.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_X, [u_far], coord_to_subsituted)
far_field_bc.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_Y, [v_far], coord_to_subsituted)

print(" ")
print("======== Printing list of conditions ========")
jumps_print = iso0_jump.getImposedConditions()
for i in jumps_print:
    print(sp.simplify(i))
jumps_print = far_field_bc.getImposedConditions()
for i in jumps_print:
    print(sp.simplify(i))
print("===============================================")

#* -------------------------------------------------------------------------- *#
#* --- 4/ PROBLEM HANDLING ---
#* -------------------------------------------------------------------------- *#

my_problem = GeneralProblemHandler([eqs_0, eqs_1], [iso0_jump, far_field_bc])
params_sol = my_problem.getParametricSolCoeff()

tag_pvT = [FluidSolutionTags.PRESSURE, FluidSolutionTags.VELOCITY_X, FluidSolutionTags.VELOCITY_Y, FluidSolutionTags.TEMPERATURE]

# # !WARNING: to be generalized
splitting_2 = False
# if m_dot != 0.0 and not splitting_2:
#     tmp_params_sym = list(params_sol.keys())
#     # print(params_sol[tmp_params_sym[0]].subs({tmp_params_sym[0]: -200.252}))
#     # params_sol[tmp_params_sym[0]] = params_sol[tmp_params_sym[0]].subs({tmp_params_sym[0]: -200.252})
#     bp_1_symbol = tmp_params_sym[0]
#     bP_1_sols = list(sp.solve(params_sol[bp_1_symbol]-bp_1_symbol, bp_1_symbol))
#     bP_1_value = min(bP_1_sols, key=abs)#-200.252
#     print("======= Pressure coeff value for m_dot != 0: %2.12f ========"%bP_1_value)
#     for key, value in params_sol.items() :
#         params_sol[key] = value.subs({bp_1_symbol: bP_1_value})

print(" ")
print("======== Printing values of unknown parameters ========")
for key, value in params_sol.items() :
    print ("%s =  %s"%(str(key), str(value)))
print("========================================================")

print(" ")
print("======== Printing (p, v, T) solutions vectors of both phases ========")
print("*** Phase 0 ***")
for i in tag_pvT:
    print(my_problem.getThisPhysicalModel(name_model_0).getSolTag(i, params_sol))
print("*** Phase 1 ***")
for i in tag_pvT:
    print(my_problem.getThisPhysicalModel(name_model_1).getSolTag(i, params_sol))
print(" ")
print("======== Printing normal to boundaries ========")
print(my_problem.getThisBoundary("iso0").getBoundaryGeometry().getNormalToBoundary())

#* -------------------------------------------------------------------------- *#
#* --- 5/ CHANGE OF COORDINATES SYSTEM (OPTIONAL!) ---
#* -------------------------------------------------------------------------- *#
new_sym_variables = sp.symbols('x, y')
center_new_domain = [1.0, 1.0]
my_new_problem = my_problem.copyAndChangeCoordinatesSystem(list(new_sym_variables), CoordinatesSystemType.CARTESIAN, center_new_domain, [[2,0.0]])
params_sol_new = my_new_problem.getParametricSolCoeff()

print(" ")
print("======== Printing (p, v, T) solutions vectors in new Cartesian reference frame ========")
print("*** Phase 0 ***")
for i in tag_pvT:
    print(my_new_problem.getThisPhysicalModel(name_model_0).getSolTag(i, params_sol_new))
print("*** Phase 1 ***")
for i in tag_pvT:
    print(my_new_problem.getThisPhysicalModel(name_model_1).getSolTag(i, params_sol_new))

print(" ")
print("======== Printing normal to boundaries ========")
print(my_new_problem.getThisBoundary("iso0").getBoundaryGeometry().getBoundaryEquation())
new_normal = my_new_problem.getThisBoundary("iso0").getBoundaryGeometry().getNormalToBoundary()
print([sp.simplify(i) for i in new_normal])

#* -------------------------------------------------------------------------- *#
#* --- 6/ OUTPUTS PRINTING AND PLOTS ---
#* -------------------------------------------------------------------------- *#
# Print in files
output_path = output_absolute_path
my_new_problem.printMMSSourceTermInFile(output_path, OutputFileType.TEXT)
my_new_problem.printSolutionVectorInFile([], CompressibleFlowVarSetTags.PRIMITIVE_PVT, output_path, OutputFileType.TEXT)

tags_var_to_print = [FluidSolutionTags.PRESSURE, FluidSolutionTags.VELOCITY_X, FluidSolutionTags.VELOCITY_Y, FluidSolutionTags.TEMPERATURE,
                     FluidSolutionTags.DENSITY, FluidSolutionTags.MOMENTUM_X, FluidSolutionTags.MOMENTUM_Y, FluidSolutionTags.RHOTOTALENERGY,
                     FluidSolutionGradientTags.SHEARSTRESS_XX, FluidSolutionGradientTags.SHEARSTRESS_XY, FluidSolutionGradientTags.SHEARSTRESS_YY,
                     FluidSolutionGradientTags.HEATFLUX_X, FluidSolutionGradientTags.HEATFLUX_Y, FluidSolutionGradientTags.VORTICITY]

# tags_var_to_print = [FluidSolutionTags.CONVECTIVEFLUX_X1, FluidSolutionTags.CONVECTIVEFLUX_X2, FluidSolutionTags.CONVECTIVEFLUX_X3, FluidSolutionTags.CONVECTIVEFLUX_X4,
#                      FluidSolutionTags.CONVECTIVEFLUX_Y1, FluidSolutionTags.CONVECTIVEFLUX_Y2, FluidSolutionTags.CONVECTIVEFLUX_Y3, FluidSolutionTags.CONVECTIVEFLUX_Y4,
#                      FluidSolutionGradientTags.DIFFUSIVEFLUX_X1, FluidSolutionGradientTags.DIFFUSIVEFLUX_X2, FluidSolutionGradientTags.DIFFUSIVEFLUX_X3, FluidSolutionGradientTags.DIFFUSIVEFLUX_X4,
#                      FluidSolutionGradientTags.DIFFUSIVEFLUX_Y1, FluidSolutionGradientTags.DIFFUSIVEFLUX_Y2, FluidSolutionGradientTags.DIFFUSIVEFLUX_Y3, FluidSolutionGradientTags.DIFFUSIVEFLUX_Y4]

my_new_problem.printAnyFieldDataInFile(tags_var_to_print, output_path, OutputFileType.TEXT)

# Area over which to plot
plot_area_cylind = PlotOver2DCircularAreaCenteredAtOrigin(sym_variables, [my_problem.getThisBoundary("iso0").getBoundaryGeometry()], R_far)

R_far = 0.5
pt_1 = [center_new_domain[0]-R_far, center_new_domain[1]-R_far]
pt_2 = [center_new_domain[0]-R_far, center_new_domain[1]+R_far]
pt_3 = [center_new_domain[0]+R_far, center_new_domain[1]+R_far]
pt_4 = [center_new_domain[0]+R_far, center_new_domain[1]-R_far]
plot_area_cart = PlotOver2DAreaWithStraightBoundaries(new_sym_variables, [my_new_problem.getThisBoundary("iso0").getBoundaryGeometry()], [pt_1,pt_2,pt_3,pt_4])

# 1-D lines along which to plot
nb_lines_plot = 4
coords_plot = np.linspace(0.0, 360.0, nb_lines_plot)
lines_plot = []
for i in range(nb_lines_plot):
    lines_plot.append(BoundaryGeometryFromEquation(sym_variables, [0.0, 1.0, coords_plot[i]], "plot_line_"+str(i)))

# nb_lines_plot = 4
# coords_plot = np.linspace(center_new_domain[1]-radius, center_new_domain[1]+radius, nb_lines_plot)
# lines_plot = []
# for i in range(nb_lines_plot):
#     lines_plot.append(BoundaryGeometryFromEquation(new_sym_variables, [0.0, 1.0, coords_plot[i]], "plot_line_"+str(i)))

# Quantities to plot
Fcx4_plot = QuantityInfoForPlot(FluidSolutionTags.CONVECTIVEFLUX_X4)
Fcy4_plot = QuantityInfoForPlot(FluidSolutionTags.CONVECTIVEFLUX_Y4)
rhoE_plot = QuantityInfoForPlot(FluidSolutionTags.RHOTOTALENERGY)
u_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_X)
v_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_Y)
vel_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_VEC, False, ProjectionType.NOPROJECTION, None, True)
vel_n_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_VEC, False, ProjectionType.NORMAL)
vel_t_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_VEC, False, ProjectionType.TANGENT)
T_plot = QuantityInfoForPlot(FluidSolutionTags.TEMPERATURE)
p_plot = QuantityInfoForPlot(FluidSolutionTags.PRESSURE)
rho_plot = QuantityInfoForPlot(FluidSolutionTags.DENSITY)
tau_nn_plot = QuantityInfoForPlot(FluidSolutionGradientTags.SHEARSTRESS, True, ProjectionType.NORMALNORMAL)
tau_nt_plot = QuantityInfoForPlot(FluidSolutionGradientTags.SHEARSTRESS, True, ProjectionType.NORMALTANGENT)
q_n_plot = QuantityInfoForPlot(FluidSolutionGradientTags.HEATFLUX, True, ProjectionType.NORMAL)
q_minus_tauu_n_plot = QuantityInfoForPlot(FluidSolutionGradientTags.HEATFLUX_MINUS_VISCOUSDISSIPATION, True, ProjectionType.NORMAL)
quantities_plot = [vel_plot, u_plot, v_plot, vel_n_plot, vel_t_plot, T_plot, p_plot, rho_plot, tau_nn_plot, tau_nt_plot, q_n_plot, q_minus_tauu_n_plot]

# Sides of interface
side_interface_plot = dict()
side_interface_plot[name_model_0] = -1
side_interface_plot[name_model_1] = 1

# Plot everything
my_problem.plotQuantitiesAlongOneDimLinesOverThisArea(quantities_plot, lines_plot, plot_area_cylind, side_interface_plot, 20, 0)
my_new_problem.plotQuantitiesOverThisTwoDimArea(quantities_plot, plot_area_cart, side_interface_plot, 20)

