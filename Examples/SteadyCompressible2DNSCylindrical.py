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


output_folder_name = "SteadyCompressible2DNSCylindrical/"
os.chdir(script_path)
output_absolute_path = os.path.join(script_path,output_folder_name)

#* -------------------------------------------------------------------------- *#
#* --- 0/ DEFINITION OF PROBLEM PARAMETERS ---
#* -------------------------------------------------------------------------- *#
do_plot = True
# Domain geometry
Lx = [-0.5,0.5]
Ly = [-0.5,0.5]
R_middle_cylinder = 0.1
domain_dim = 2

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

# Boundary conditions values
p_far_field = 1.1e5
u_cylinder = 0.0
v_cylinder = 0.0
T_cylinder = 300
sol_dependent_on_theta = False
stiff_gradients = False

#* -------------------------------------------------------------------------- *#
#* --- 1/ MANUFACTURED SOLUTION SPECIFICATION ---
#* -------------------------------------------------------------------------- *#
sym_variables = sp.symbols('r, theta', real=True)
r = sym_variables[0]
theta = sym_variables[1]

# if stiff_gradients:

if stiff_gradients:
    A_p = 10000.0
    A_u = 5.0
    k_u_theta = 1.0
    k_u_r = 10.0
    A_v = 5.0
    k_v_theta = 1.0
    k_v_r = 10.0
    A_T = 10.0
    k_T_theta = 1.0
    k_T_r = 6.0
    power_p = 4
else:
    A_p = 1.0
    A_u = 1.0
    k_u_theta = 1.0
    k_u_r = 1.0
    A_v = 1.0
    k_v_theta = 1.0
    k_v_r = 1.0
    A_T = 2.0
    k_T_theta = 1.0
    k_T_r = 3.0
    power_p = 2


B_p = [sp.symbols('B_p')] 
B_u = [sp.symbols('B_u')] 
B_v = [sp.symbols('B_v')] 
B_T = [sp.symbols('B_T')] 

if not sol_dependent_on_theta:
    k_u_theta = k_v_theta = k_T_theta = 0.0

# Pressure
if stiff_gradients:
    p_expression = A_p*sp.Pow(r,power_p)+B_p[0]
else:
    p_expression = A_p*sp.cos(3.0*r)+B_p[0]
p = UserDefinedManufSol(B_p, sym_variables, p_expression)

# Velocities
u_expression = A_u*sp.cos(k_u_theta*theta)*sp.sin(k_u_r*r)+B_u[0]
v_expression = A_v*sp.cos(k_v_theta*theta)*sp.sin(-k_v_r*r)+B_v[0]
u = UserDefinedManufSol(B_u, sym_variables, u_expression)
v = UserDefinedManufSol(B_v, sym_variables, v_expression)

#Temperature
T_expression = A_T*sp.cos(k_T_theta*theta)*sp.sin(-k_T_r*r)+B_T[0]
T = UserDefinedManufSol(B_T, sym_variables, T_expression)

manuf_sol_list = [[p], [u, v], [T]]
manuf_sol_cont = CompressibleFlowManufSolContainer(manuf_sol_list, domain_dim, sym_variables, var_choice, [])

#* -------------------------------------------------------------------------- *#
#* --- 2/ DEFINITION OF THE GOVERNING EQUATIONS ---
#* -------------------------------------------------------------------------- *#

name_model = "NSeqs"
eqs = CompressibleNavierStokesModels(name_model, manuf_sol_cont, domain_dim, EOS, transp_prop, CoordinatesSystemType.CYLINDRICAL)

#* -------------------------------------------------------------------------- *#
#* --- 3/ DEFINITION OF THE BOUNDARY  ---
#* -------------------------------------------------------------------------- *#

# Definition of boundaries geometry
cylinder_geo = BoundaryGeometryFromEquation(sym_variables, [1.0, 0.0, R_middle_cylinder], "cylinder")
far_field_geo = BoundaryGeometryFromEquation(sym_variables, [1.0, 0.0, Lx[1]], "farField")
coord_to_subsituted = sym_variables[0]

# Definition of BC on the cylinder wall
cylinder_bc = GeneralBoundaryConditions([eqs], cylinder_geo)
cylinder_bc.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_X, [u_cylinder], coord_to_subsituted)
cylinder_bc.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.VELOCITY_Y, [v_cylinder], coord_to_subsituted)
cylinder_bc.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.TEMPERATURE, [T_cylinder], coord_to_subsituted)

# Definition of BC on the far field wall
far_field_bc = GeneralBoundaryConditions([eqs], far_field_geo)
far_field_bc.addDirichletBCAndSubsituteBoundaryEq(FluidSolutionTags.PRESSURE, [p_far_field], coord_to_subsituted)

print(" ")
print("======== Printing list of conditions ========")
bc_print = cylinder_bc.getImposedConditions()
for i in bc_print:
    print(sp.simplify(i))
bc_print = far_field_bc.getImposedConditions()
for i in bc_print:
    print(sp.simplify(i))

#* -------------------------------------------------------------------------- *#
#* --- 4/ PROBLEM HANDLING ---
#* -------------------------------------------------------------------------- *#

my_problem = GeneralProblemHandler([eqs], [cylinder_bc, far_field_bc])
params_sol = my_problem.getParametricSolCoeff()

tag_sols = [FluidSolutionTags.PRESSURE, FluidSolutionTags.VELOCITY_X, FluidSolutionTags.VELOCITY_Y, FluidSolutionTags.TEMPERATURE, FluidSolutionTags.DENSITY]

print(" ")
print("======== Printing values of unknown parameters ========")
for key, value in params_sol.items() :
    print ("%s =  %s"%(str(key), str(value)))
print("========================================================")

print(" ")
print("======== Printing variables expressions in cylindrical coordinates ========")
for i in tag_sols:
    print("")
    print(str(i))
    print(my_problem.getThisPhysicalModel(name_model).getSolTag(i, params_sol))

#* -------------------------------------------------------------------------- *#
#* --- 5/ CHANGE OF COORDINATES SYSTEM (from cylinderical to cartesian) ---
#* -------------------------------------------------------------------------- *#
new_sym_variables = sp.symbols('x, y')
center_new_domain = [0.5*(Lx[0]+Lx[1]), 0.5*(Ly[0]+Ly[1])]
my_new_problem = my_problem.copyAndChangeCoordinatesSystem(list(new_sym_variables), CoordinatesSystemType.CARTESIAN, center_new_domain, None)
params_sol_new = my_new_problem.getParametricSolCoeff()

print(" ")
print("======== Printing variables expressions in Cartesian coordinates ========")
for i in tag_sols:
    print("")
    print(str(i))
    print(my_new_problem.getThisPhysicalModel(name_model).getSolTag(i, params_sol_new))

#* -------------------------------------------------------------------------- *#
#* --- 6/ OUTPUTS PRINTING AND PLOTS ---
#* -------------------------------------------------------------------------- *#

# Print in files
output_path = output_absolute_path
my_new_problem.printMMSSourceTermInFile(output_path, OutputFileType.TEXT, False, True)
my_new_problem.printSolutionVectorInFile([], var_choice, output_path, OutputFileType.TEXT)

if do_plot:
    # Area over which to plot
    pt_1 = [Lx[0], Ly[0]]
    pt_2 = [Lx[0], Ly[1]]
    pt_3 = [Lx[1], Ly[1]]
    pt_4 = [Lx[1], Ly[0]]

    plot_area_cart = PlotOver2DAreaWithStraightBoundaries(new_sym_variables, [my_new_problem.getThisBoundary("cylinder").getBoundaryGeometry()], [pt_1, pt_2, pt_3, pt_4])

        # 1-D lines along which to plot
    nb_lines_plot = 4
    coords_plot = np.linspace(0.0, 360.0, nb_lines_plot)
    lines_plot = []
    for i in range(nb_lines_plot):
        lines_plot.append(BoundaryGeometryFromEquation(sym_variables, [0.0, 1.0, coords_plot[i]], "plot_line_"+str(i)))

    # Quantities to plot
    u_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_X)
    v_plot = QuantityInfoForPlot(FluidSolutionTags.VELOCITY_Y)
    T_plot = QuantityInfoForPlot(FluidSolutionTags.TEMPERATURE)
    p_plot = QuantityInfoForPlot(FluidSolutionTags.PRESSURE)
    rho_plot = QuantityInfoForPlot(FluidSolutionTags.DENSITY)
    src_conv_plot = QuantityInfoForPlot(MMSSourceTermTags.MMS_SOURCE_DIFFUSIVE_OVER_CONVECTIVE, False)

    quantities_plot = [T_plot, u_plot, v_plot, p_plot, rho_plot] #src_conv_plot

    # Sides of interface
    side_interface_plot = dict()
    side_interface_plot[name_model] = 1

    # Plot everything

    # my_problem.plotQuantitiesAlongOneDimLinesOverThisArea(quantities_plot, lines_plot, plot_area_cart, side_interface_plot, 20, 0)

    my_new_problem.plotQuantitiesOverThisTwoDimArea(quantities_plot, plot_area_cart, side_interface_plot, 30)

