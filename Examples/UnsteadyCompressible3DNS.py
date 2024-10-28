import os
script_path = os.path.dirname(os.path.realpath(__file__))
import sys
project_path = script_path+"/../."
sys.path.insert(0, project_path)

import sympy as sp
from PhysicalModels.ThermoPhysicalModels.EOS_factory import PerfectGasEOS
from PhysicalModels.ThermoPhysicalModels.TransportProperties_factory import ConstantFluidTransportProperty
from PhysicalModels.CompressibleNavierStokesModels import CompressibleNavierStokesModels
from ManufSolution.ManufSolShape.PolarPolynomialManufSol import PolarPolynomialManufSol
from ManufSolution.ManufSolShape.UserDefinedManufSol import UserDefinedManufSol
from ManufSolution.ManufSolContainer.CompressibleFlowManufSolContainer import CompressibleFlowManufSolContainer
from Common.MMSTags import CompressibleFlowVarSetTags, MMSSourceTermTags, FluidSolutionTags, FluidSolutionGradientTags, ProjectionType
from BoundaryConditions.BoundaryGeometry import BoundaryGeometryFromEquation, TimeDependentBoundaryGeometryFromEquation
from BoundaryConditions.GeneralBoundaryConditions import GeneralBoundaryConditions
from Common.MMSTags import CoordinatesSystemType, OutputFileType
from ProblemSolving.GeneralProblemHandler import GeneralProblemHandler, QuantityInfoForPlot
from Outputs.PlotOverArea import *
import copy


output_folder_name = "UnsteadyCompressible3DNS/"
os.chdir(script_path)
output_absolute_path = os.path.join(script_path,output_folder_name)

#* -------------------------------------------------------------------------- *#
#* --- 0/ DEFINITION OF PROBLEM PARAMETERS ---
#* -------------------------------------------------------------------------- *#
do_plot = True
# Domain geometry
# Lx = [0.0,1.0]
# Ly = [0.0,1.0]
# Lz = [0.0,1.0]
Lx = [-0.5,0.5]
Ly = [-0.5,0.5]
Lz = [-0.5,0.5]
domain_dim = 2
time_dependent = False

# Equation of state
gamma = 1.4
Cv = 715
EOS = PerfectGasEOS(gamma, Cv, domain_dim)

# Transport properties
dynamic_viscosity = 1.0
thermal_conductivity = 1.0
transp_prop = ConstantFluidTransportProperty(dynamic_viscosity, thermal_conductivity)

# Variables choice
var_choice = CompressibleFlowVarSetTags.PRIMITIVE_PVT


x_dim = Lx[1]-Lx[0]
y_dim = Ly[1]-Ly[0]
z_dim = Lz[1]-Lz[0]
k_x = 2.0*sp.pi/x_dim
k_y = 2.0*sp.pi/y_dim
k_z = 2.0*sp.pi/z_dim

#* -------------------------------------------------------------------------- *#
#* --- 1/ MANUFACTURED SOLUTION SPECIFICATION ---
#* -------------------------------------------------------------------------- *#
if domain_dim == 2:
    sym_variables = sp.symbols('x, y, t', real=True)
elif domain_dim == 3:
    sym_variables = sp.symbols('x, y, z, t', real=True)
else:
    raise ValueError("The domain dimension must be 2 or 3.")

A = 1.0
B = 1.0
C = 1.0
P_0 = 10000.0
T_0 = 300.0
Amp_p = 10.0
Amp_T = 10.0
Amp_vel = 10.0
omega = 10.0
nu = 0.1
if not time_dependent:
    nu = omega = 0.0

x = sym_variables[0]
y = sym_variables[1]
t = sym_variables[-1]
if domain_dim == 2:
    # Pressure
    p_expression = P_0+Amp_p*(B*C*sp.cos(k_x*x)*sp.sin(k_y*y)) * sp.cos(omega * t) * sp.exp(-2*nu * sp.Pow(k_x,2) * t)
    # p_expression = P_0*(B*C*sp.cos(k_x*x)*sp.sin(k_y*y)) * sp.exp(-2*nu * sp.Pow(k_x,2) * t)
    p = UserDefinedManufSol([], sym_variables, p_expression)

    # Velocities
    u_expression = Amp_vel*(C*sp.cos(k_y*y)) * sp.exp(-nu * sp.Pow(k_x,2) * t)
    v_expression = Amp_vel*(B*sp.sin(k_x*x)) * sp.exp(-nu * sp.Pow(k_y,2) * t)
    u = UserDefinedManufSol([], sym_variables, u_expression)
    v = UserDefinedManufSol([], sym_variables, v_expression)

    #Temperature
    # T_expression = T_0*(B*C*sp.sin(k_x*x)*sp.cos(k_y*y)) * sp.exp(-2*nu * sp.Pow(k_x,2) * t)
    T_expression = T_0+Amp_T*(B*C*sp.sin(k_x*x)*sp.cos(k_y*y)) * sp.cos(3/2*omega * t) * sp.exp(-2*nu * sp.Pow(k_x,2) * t)
    T = UserDefinedManufSol([], sym_variables, T_expression)

    manuf_sol_list = [[p], [u, v], [T]]
elif domain_dim == 3:
    z = sym_variables[2]
    # Pressure
    p_expression = P_0+Amp_p*(B*C*sp.cos(k_x*x)*sp.sin(k_y*y) + A*B*sp.sin(k_x*x)*sp.cos(k_z*z) + A*C*sp.sin(k_z*z)*sp.cos(k_y*y)) * sp.cos(omega * t) * sp.exp(-2*nu * sp.Pow(k_x,2) * t)
    p = UserDefinedManufSol([], sym_variables, p_expression)

    # Velocities
    u_expression = Amp_vel*(A*sp.sin(k_z*z) + C*sp.cos(k_y*y)) * sp.exp(-nu * sp.Pow(k_x,2) * t)
    v_expression = Amp_vel*(B*sp.sin(k_x*x) + A*sp.cos(k_z*z)) * sp.exp(-nu * sp.Pow(k_y,2) * t)
    w_expression = Amp_vel*(C*sp.sin(k_y*y) + B*sp.cos(k_x*x)) * sp.exp(-nu * sp.Pow(k_z,2) * t)
    u = UserDefinedManufSol([], sym_variables, u_expression)
    v = UserDefinedManufSol([], sym_variables, v_expression)
    w = UserDefinedManufSol([], sym_variables, w_expression)

    #Temperature
    T_expression = T_0+Amp_T*(B*C*sp.sin(k_x*x)*sp.cos(k_y*y) + A*B*sp.cos(k_x*x)*sp.sin(k_z*z) + A*C*sp.cos(k_z*z)*sp.sin(k_y*y)) *  sp.cos(3/2*omega * t) * sp.exp(-2*nu * sp.Pow(k_x,2) * t)
    T = UserDefinedManufSol([], sym_variables, T_expression)

    manuf_sol_list = [[p], [u, v, w], [T]]


time_var = []
if time_dependent:
    time_var = [t]
manuf_sol_cont = CompressibleFlowManufSolContainer(manuf_sol_list, domain_dim, sym_variables[0:domain_dim], var_choice, time_var)

#* -------------------------------------------------------------------------- *#
#* --- 2/ DEFINITION OF THE GOVERNING EQUATIONS ---
#* -------------------------------------------------------------------------- *#

name_model = "NSeqs"
eqs = CompressibleNavierStokesModels(name_model, manuf_sol_cont, domain_dim, EOS, transp_prop, CoordinatesSystemType.CARTESIAN)


#* -------------------------------------------------------------------------- *#
#* --- 3/ PROBLEM HANDLING ---
#* -------------------------------------------------------------------------- *#

my_problem = GeneralProblemHandler([eqs])
params_sol = my_problem.getParametricSolCoeff()

tag_sols = [FluidSolutionTags.PRESSURE, FluidSolutionTags.VELOCITY_X, FluidSolutionTags.VELOCITY_Y, FluidSolutionTags.TEMPERATURE, FluidSolutionTags.DENSITY]
print(" ")
print("======== Printing variables expressions ========")
for i in tag_sols:
    print("")
    print(str(i))
    print(my_problem.getThisPhysicalModel(name_model).getSolTag(i))


#* -------------------------------------------------------------------------- *#
#* --- 4/ OUTPUTS PRINTING AND PLOTS ---
#* -------------------------------------------------------------------------- *#

# Print in files
output_path = output_absolute_path
my_problem.printMMSSourceTermInFile(output_path, OutputFileType.TEXT, False, True)
my_problem.printSolutionVectorInFile([], var_choice, output_path)

if domain_dim != 2 and do_plot:
    raise ValueError("Plotting features in 3D not yet available...")

if domain_dim == 2 and do_plot:
    # Area over which to plot
    if time_dependent:
        time_plot_vec = [0.0,0.1,0.2,0.3]#0.5
    else:
        time_plot_vec = [0.0]
    pt_1 = [Lx[0], Ly[0]]
    pt_2 = [Lx[0], Ly[1]]
    pt_3 = [Lx[1], Ly[1]]
    pt_4 = [Lx[1], Ly[0]]


    for time_plot in time_plot_vec:

        print("===== TIME %2.3f ====="%time_plot)
        plot_area_cart = PlotOver2DAreaWithStraightBoundaries(sym_variables[0:domain_dim], 0, [pt_1, pt_2, pt_3, pt_4])

        # 1-D lines along which to plot
        nb_lines_plot = 2
        coords_plot = np.linspace(Lx[0]+0.01, Lx[1]-0.01, nb_lines_plot)
        lines_plot = []
        for i in range(nb_lines_plot):
            lines_plot.append(BoundaryGeometryFromEquation(sym_variables[0:domain_dim], [1.0, 0.0, coords_plot[i]], "plot_line_"+str(i)))

        # Quantities to plot
        if not time_dependent:
            time_plot = None
        u_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_X, False, ProjectionType.NOPROJECTION, time_plot)
        v_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_Y, False, ProjectionType.NOPROJECTION, time_plot)
        T_plot = QuantityInfoForPlot(FluidSolutionTags.TEMPERATURE, False, ProjectionType.NOPROJECTION, time_plot)
        p_plot = QuantityInfoForPlot(FluidSolutionTags.PRESSURE, False, ProjectionType.NOPROJECTION, time_plot)
        rho_plot = QuantityInfoForPlot(FluidSolutionTags.DENSITY, False, ProjectionType.NOPROJECTION, time_plot)
        tau_nn_plot = QuantityInfoForPlot(FluidSolutionGradientTags.SHEARSTRESS, True, ProjectionType.NORMALNORMAL, time_plot)
        tau_nt_plot = QuantityInfoForPlot(FluidSolutionGradientTags.SHEARSTRESS, True, ProjectionType.NORMALTANGENT, time_plot)
        q_n_plot = QuantityInfoForPlot(FluidSolutionGradientTags.HEATFLUX, True, ProjectionType.NORMAL, time_plot)
        q_minus_tauu_n_plot = QuantityInfoForPlot(FluidSolutionGradientTags.HEATFLUX_MINUS_VISCOUSDISSIPATION, True, ProjectionType.NORMAL, time_plot)
        
        # src_conv_plot = QuantityInfoForPlot(MMSSourceTermTags.MMS_SOURCE_DIFFUSIVE_OVER_CONVECTIVE, False, ProjectionType.NOPROJECTION, time_plot)
        src_conv_plot = QuantityInfoForPlot(MMSSourceTermTags.MMS_SOURCE_CONVECTIVE, False, ProjectionType.NOPROJECTION, time_plot)
        src_diff_plot = QuantityInfoForPlot(MMSSourceTermTags.MMS_SOURCE_DIFFUSIVE, False, ProjectionType.NOPROJECTION, time_plot)

        quantities_plot = [src_conv_plot, src_diff_plot]
        # quantities_plot = [T_plot, p_plot, u_plot, v_plot, rho_plot]

        # Sides of interface
        side_interface_plot = dict()
        side_interface_plot[name_model] = -1

        # Plot everything

        # my_problem.plotQuantitiesAlongOneDimLinesOverThisArea(quantities_plot, lines_plot, plot_area_cart, side_interface_plot, 20, 1)

        my_problem.plotQuantitiesOverThisTwoDimArea(quantities_plot, plot_area_cart, side_interface_plot, 20)

