import os
script_path = os.path.dirname(os.path.realpath(__file__))
import sys
project_path = script_path+"/../."
sys.path.insert(0, project_path)

import sympy as sp
import copy

# --- MODULE with general utilities ---
from Common.MMSTags import CompressibleFlowVarSetTags, FluidSolutionTags, FluidSolutionGradientTags, ProjectionType
from Common.MMSTags import CoordinatesSystemType, OutputFileType

# --- MODULE 1: ManufSolution ---
from ManufSolution.ManufSolShape.UserDefinedManufSol import UserDefinedManufSol
from ManufSolution.ManufSolContainer.CompressibleFlowManufSolContainer import CompressibleFlowManufSolContainer

# --- MODULE 2: PhysicalModels ---
from PhysicalModels.ThermoPhysicalModels.EOS_factory import StiffenedGasEOS
from PhysicalModels.ThermoPhysicalModels.TransportProperties_factory import ConstantFluidTransportProperty
from PhysicalModels.CompressibleNavierStokesModels import CompressibleNavierStokesModels

# --- MODULE 3: BoundaryConditions ---
from BoundaryConditions.BoundaryGeometry import BoundaryGeometryFromEquation, TimeDependentBoundaryGeometryFromEquation
from BoundaryConditions.GeneralBoundaryConditions import GeneralBoundaryConditions

# --- MODULE 4: ProblemSolving ---
from ProblemSolving.GeneralProblemHandler import GeneralProblemHandler, QuantityInfoForPlot

# --- MODULE 5: Outputs ---
from Outputs.PlotOverArea import *


current_directory = os.getcwd()
output_folder_name = "PAPER_UnsteadyInclinedInterfaceOutput/"
output_absolute_path = os.path.join(current_directory,output_folder_name)
output_path = output_absolute_path

#* -------------------------------------------------------------------------- *#
#* --- 0/ DEFINITION OF PROBLEM PARAMETERS ---
#* -------------------------------------------------------------------------- *#
# Domain geometry
L = [0.0,1.0]
H = [0.0,1.0]
x_dim = L[1]-L[0]
y_dim = H[1]-H[0]
y_interface_pos = H[0] + y_dim/3
domain_dim = 2

# Jumps conditions parameters
m_dot = 0.0 
deltaTangentStress = 0.0
deltaTangentVel = -0.1
delta_p = -0.2
deltaTemp = 0.5
heatSource = 1.0

#* -------------------------------------------------------------------------- *#
#* --- 1/ MANUFACTURED SOLUTION ASSEMBLY ---
#* -------------------------------------------------------------------------- *#
sym_variables = sp.symbols('X, Y, t')

# Temperatures
T_0_user = 1.0*sp.cos(0.75*sp.pi*sym_variables[1]) + 2.0
T_0 = UserDefinedManufSol([], sym_variables, T_0_user)

T_coeff = list(sp.symbols('bT_1, bT_2'))
T_1_user = 0.1*T_coeff[0]*sp.cos(5*sp.pi*sym_variables[1]/(4*y_dim)) + T_coeff[1]
T_1 = UserDefinedManufSol(T_coeff, sym_variables, T_1_user)

# Pressures
p_0_user = 1.0*sp.cos(0.75*sp.pi*sym_variables[1]) + 2.45
p_0 = UserDefinedManufSol([], sym_variables, p_0_user)

p_coeff = [sp.symbols('bP_1')]
p_1_user = 1.0*sp.cos(5*sp.pi*sym_variables[1]/(4*y_dim)) + p_coeff[0]
p_1 = UserDefinedManufSol(p_coeff, sym_variables, p_1_user)

# Horizontal velocities
u_0_user = 0.0*sym_variables[1]+0.0
u_0 = UserDefinedManufSol([], sym_variables, u_0_user)

u_coeff = list(sp.symbols('bU_1, bU_2'))
u_1_user = u_coeff[0]*sp.cos(5*sp.pi*sym_variables[1]/(4*y_dim)) + u_coeff[1] + 0.0
u_1 = UserDefinedManufSol(u_coeff, sym_variables, u_1_user)

# Vertical velocities
v_0_user = 0.0*sym_variables[1]+1.0
v_0 = UserDefinedManufSol([], sym_variables, v_0_user)

v_coeff = list(sp.symbols('bV_1, bV_2'))
v_1_user = v_coeff[0]*sp.cos(5*sp.pi*sym_variables[1]/(4*y_dim)) + v_coeff[1] + 0.0
v_1 = UserDefinedManufSol(v_coeff, sym_variables, v_1_user)

# Assembly of manuf sol containers
manuf_sol_list_0 = [[p_0], [u_0, v_0], [T_0]]
manuf_sol_list_1 = [[p_1], [u_1, v_1], [T_1]]
manuf_sol_cont_0 = CompressibleFlowManufSolContainer(manuf_sol_list_0, domain_dim, sym_variables[0:2], CompressibleFlowVarSetTags.PRIMITIVE_PVT, [sym_variables[-1]])
manuf_sol_cont_1 = CompressibleFlowManufSolContainer(manuf_sol_list_1, domain_dim, sym_variables[0:2], CompressibleFlowVarSetTags.PRIMITIVE_PVT, [sym_variables[-1]])

#* -------------------------------------------------------------------------- *#
#* --- 2/ DEFINITION OF THE GOVERNING EQUATIONS ---
#* -------------------------------------------------------------------------- *#

# Equation of states
gamma_0 = 1.4
Cv_0 = 2.5
p_inf_0 = 0.0
EOS_phase_0 = StiffenedGasEOS(gamma_0, Cv_0, p_inf_0, domain_dim)

gamma_1 = 1.6
Cv_1 = 2.0
p_inf_1 = 0.0
EOS_phase_1 = StiffenedGasEOS(gamma_1, Cv_1, p_inf_1, domain_dim) 

# Transport properties
mu_0 = 1.0
lambda_0 = 1.0
transp_prop_0 = ConstantFluidTransportProperty(mu_0, lambda_0)

mu_1 = 0.25
lambda_1 = 0.25
transp_prop_1 = ConstantFluidTransportProperty(mu_1, lambda_1)

# Equations models
name_model_0 = "phase0"
eqs_0 = CompressibleNavierStokesModels(name_model_0, manuf_sol_cont_0, domain_dim, EOS_phase_0, transp_prop_0, CoordinatesSystemType.CARTESIAN)

name_model_1 = "phase1"
eqs_1 = CompressibleNavierStokesModels(name_model_1, manuf_sol_cont_1, domain_dim, EOS_phase_1, transp_prop_1, CoordinatesSystemType.CARTESIAN)

#* -------------------------------------------------------------------------- *#
#* --- 3/ DEFINITION OF THE JUMPS CONDITIONS ---
#* -------------------------------------------------------------------------- *#

# Definition of interface geometry
vel_interface = 1.0
iso0_geo = TimeDependentBoundaryGeometryFromEquation(list(sym_variables[0:2]), [0.0, -1.0, -y_interface_pos], "iso0", [sym_variables[-1]], [vel_interface])
iso0_geo_bis = TimeDependentBoundaryGeometryFromEquation(list(sym_variables[0:2]), [0.0, -1.0, -y_interface_pos], "iso0_bis", [sym_variables[-1]], [vel_interface])
coord_to_subsituted = sym_variables[1]

# === Jumps on interface ===
iso0_jump = GeneralBoundaryConditions([eqs_0, eqs_1], iso0_geo)
iso0_jump_bis = GeneralBoundaryConditions([eqs_1], iso0_geo_bis)

# -> Dirichlet jumps on solutions

# - Normal velocity
jump_v_n = 0.0
rho_0 = eqs_0.getSolTag(FluidSolutionTags.DENSITY)
rho_1 = eqs_1.getSolTag(FluidSolutionTags.DENSITY)
if m_dot != 0.0:
    jump_inv_rho = ((1.0/rho_0) - (1.0/rho_1))
    jump_v_n = m_dot * jump_inv_rho
iso0_jump_bis.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_VEC, [-vel_interface], coord_to_subsituted, ProjectionType.NORMAL)

# - Tangential velocity
iso0_jump_bis.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_VEC, [deltaTangentVel], coord_to_subsituted, ProjectionType.TANGENT)

# - Temperature
iso0_jump.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.TEMPERATURE, [deltaTemp], coord_to_subsituted)

# - Pressure
jump_pressure = delta_p - m_dot * jump_v_n
iso0_jump.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.PRESSURE, [jump_pressure], coord_to_subsituted)

# -> Neumann jumps on fluxes

# - Normal shear stress
p_0 = eqs_0.getSolTag(FluidSolutionTags.PRESSURE)
p_1 = eqs_1.getSolTag(FluidSolutionTags.PRESSURE) 
jump_tau_nn = 0.0 # -[tau_nn] = 0.0 -> [p] = sigma*kappa - m_dot * [u_n]
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
jump_q_minus_tau = -m_dot * (E_0-E_1) - (p_0*vel_0_n_gamma-p_1*vel_1_n_gamma) + delta_p * u_gamma_n + u_gamma_t * deltaTangentStress + heatSource
iso0_jump.addNeumannBCAndSubsituteBoundaryEq(FluidSolutionGradientTags.HEATFLUX_MINUS_VISCOUSDISSIPATION, [jump_q_minus_tau], coord_to_subsituted, ProjectionType.NORMAL)

#* -------------------------------------------------------------------------- *#
#* --- 4/ PROBLEM HANDLING ---
#* -------------------------------------------------------------------------- *#
# During the construction of the problem, the system to determine the unknow 
# parameters is solved
my_problem = GeneralProblemHandler([eqs_0, eqs_1], [iso0_jump, iso0_jump_bis])

#* -------------------------------------------------------------------------- *#
#* --- 5/ CHANGE OF COORDINATES SYSTEM ---
#* -------------------------------------------------------------------------- *#
new_sym_variables = sp.symbols('x, y')
angle_rotation = -15
my_new_problem = my_problem.copyAndChangeCoordinatesSystem(list(new_sym_variables), CoordinatesSystemType.CARTESIAN, None, [[2,angle_rotation]])
params_sol_new = my_new_problem.getParametricSolCoeff()

#* -------------------------------------------------------------------------- *#
#* --- 6/ COMPUTATION OF SOURCE TERMS ---
#* -------------------------------------------------------------------------- *#
# This subsitutes the parameters values into the symbolic solutions expressions and write them in a file
my_new_problem.printSolutionVectorInFile([], CompressibleFlowVarSetTags.PRIMITIVE_PVT, output_path)
# This computes the source terms and write them in a file
my_new_problem.printMMSSourceTermInFile(output_path, OutputFileType.TEXT, False, True)

#* -------------------------------------------------------------------------- *#
#* --- 7/ PLOTS ---
#* -------------------------------------------------------------------------- *#
# Area over which to plot
time_plot_vec = [0.0,0.2]
pt_1 = [L[0], H[0]]
pt_2 = [L[0], H[1]]
pt_3 = [L[1], H[1]]
pt_4 = [L[1], H[0]]

iso0_plot_geo_main = my_new_problem.getThisBoundary("iso0").getBoundaryGeometry()
for time_plot in time_plot_vec:

    print("===== TIME %2.3f ====="%time_plot)

    iso0_plot_geo = copy.deepcopy(iso0_plot_geo_main)
    iso0_plot_geo.setBoundaryEquation(iso0_plot_geo.getBoundaryEquation().subs({sym_variables[-1]: time_plot}))
    iso0_plot_geo.setSymVariables(new_sym_variables)
    plot_area_cart = PlotOver2DAreaWithStraightBoundaries(new_sym_variables, [iso0_plot_geo], [pt_1, pt_2, pt_3, pt_4])

    # 1-D lines along which to plot
    nb_lines_plot = 4
    coords_plot = np.linspace(L[0]+0.01, L[1]-0.01, nb_lines_plot)
    lines_plot = []
    for i in range(nb_lines_plot):
        lines_plot.append(BoundaryGeometryFromEquation(new_sym_variables, [1.0, 0.0, coords_plot[i]], "plot_line_"+str(i)))
    lines_plot_original = []
    for i in range(nb_lines_plot):
        lines_plot_original.append(BoundaryGeometryFromEquation(sym_variables, [1.0, 0.0, coords_plot[i]], "plot_line_"+str(i)))

    # Quantities to plot
    u_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_X, False, ProjectionType.NOPROJECTION, time_plot)
    v_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_Y, False, ProjectionType.NOPROJECTION, time_plot)
    vel_n_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_VEC, False, ProjectionType.NORMAL, time_plot)
    vel_t_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_VEC, False, ProjectionType.TANGENT, time_plot)
    T_plot = QuantityInfoForPlot(FluidSolutionTags.TEMPERATURE, False, ProjectionType.NOPROJECTION, time_plot)
    p_plot = QuantityInfoForPlot(FluidSolutionTags.PRESSURE, False, ProjectionType.NOPROJECTION, time_plot)
    rho_plot = QuantityInfoForPlot(FluidSolutionTags.DENSITY, False, ProjectionType.NOPROJECTION, time_plot)
    tau_nn_plot = QuantityInfoForPlot(FluidSolutionGradientTags.SHEARSTRESS, True, ProjectionType.NORMALNORMAL, time_plot)
    tau_nt_plot = QuantityInfoForPlot(FluidSolutionGradientTags.SHEARSTRESS, True, ProjectionType.NORMALTANGENT, time_plot)
    q_n_plot = QuantityInfoForPlot(FluidSolutionGradientTags.HEATFLUX, True, ProjectionType.NORMAL, time_plot)
    q_minus_tauu_n_plot = QuantityInfoForPlot(FluidSolutionGradientTags.HEATFLUX_MINUS_VISCOUSDISSIPATION, True, ProjectionType.NORMAL, time_plot)
    # quantities_plot = [u_plot, v_plot, vel_n_plot, vel_t_plot, T_plot, p_plot, rho_plot, tau_nn_plot, tau_nt_plot, q_n_plot, q_minus_tauu_n_plot]

    # quantities_plot = [u_plot, v_plot, vel_n_plot, vel_t_plot]#, T_plot, p_plot, rho_plot, tau_nn_plot, tau_nt_plot]
    quantities_plot = [p_plot]#[vel_n_plot, vel_t_plot, u_plot, v_plot]#

    # Sides of interface
    side_interface_plot = dict()
    side_interface_plot[name_model_0] = -1
    side_interface_plot[name_model_1] = 1

    # Plot everything
    # my_problem.plotQuantitiesAlongOneDimLinesOverThisArea(quantities_plot, lines_plot_original, plot_area_cart_original, side_interface_plot, 20, 1)

    my_new_problem.plotQuantitiesAlongOneDimLinesOverThisArea(quantities_plot, lines_plot, plot_area_cart, side_interface_plot, 20, 1)

    my_new_problem.plotQuantitiesOverThisTwoDimArea(quantities_plot, plot_area_cart, side_interface_plot, 40)
