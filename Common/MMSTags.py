from enum import Enum

#* --- TAGS FOR SETS OF VARIABLES --- *#
# - Abstract level
class VarSetTags(Enum):
    pass

# - By-default variables set
class DefaultVarSetTags(VarSetTags):
    MYVARSETTAG = 1

# - Tags for Navier-Stokes fluid systems
class CompressibleFlowVarSetTags(VarSetTags):
    CONSERVATIVE = 1
    PRIMITIVE_PVT = 2
    PRIMITIVE_PVRHO = 3



#* --- TAGS FOR SOLUTION VARIABLES --- *#
# - Abstract level
class SolutionTags(Enum):
    pass

#* --- TAGS FOR MMS SOURCE TERMS --- *#
class MMSSourceTermTags(SolutionTags):
    MMS_SOURCE_CONVECTIVE = 1
    MMS_SOURCE_DIFFUSIVE = 2
    MMS_SOURCE_SRC = 3
    MMS_SOURCE_UNSTEADY = 4
    MMS_SOURCE_FULL = 5
    MMS_SOURCE_CONVECTIVE_OVER_DIFFUSIVE = 6
    MMS_SOURCE_DIFFUSIVE_OVER_CONVECTIVE = 7

# - By-default solution tags
class DefaultSolutionTags(SolutionTags):
    VAR_1 = 1
    VAR_2 = 2
    VAR_3 = 3
    VAR_4 = 4
    VAR_5 = 5
    VAR_6 = 6
    VAR_7 = 7
    VAR_8 = 8
    VAR_9 = 9
    VAR_10 = 10

# - Tags for Navier-Stokes fluid systems
class FluidSolutionTags(SolutionTags):
    DENSITY = 1
    PRESSURE = 2
    TEMPERATURE = 3
    VELOCITY_X = 4
    VELOCITY_Y = 5
    VELOCITY_Z = 6
    VELOCITY_VEC = 7
    MOMENTUM_X = 8
    MOMENTUM_Y = 9
    MOMENTUM_Z = 10
    MOMENTUM_VEC = 11
    INTERNALENERGY = 12
    KINETICENERGY = 13
    TOTALENERGY = 14
    RHOTOTALENERGY = 15
    ENTHALPY = 16
    TOTALENTHALPY = 17
    CONVECTIVEFLUX_X1 = 18
    CONVECTIVEFLUX_X2 = 19
    CONVECTIVEFLUX_X3 = 20
    CONVECTIVEFLUX_X4 = 21
    CONVECTIVEFLUX_Y1 = 22
    CONVECTIVEFLUX_Y2 = 23
    CONVECTIVEFLUX_Y3 = 24
    CONVECTIVEFLUX_Y4 = 25

#* --- TAGS FOR GRADIENT VARIABLES --- *#
# - Abstract level
class SolutionGradientTags(Enum):
    pass

# # - By-default solution tags
# class DefaultGradientSolutionTags(SolutionGradientTags):
#     GRADVAR_1 = 1
#     GRADVAR_2 = 2
#     GRADVAR_3 = 3
#     GRADVAR_4 = 4
#     GRADVAR_5 = 5
#     GRADVAR_6 = 6
#     GRADVAR_7 = 7
#     GRADVAR_8 = 8
#     GRADVAR_9 = 9
#     GRADVAR_10 = 10

# - Tags for Navier-Stokes fluid systems
class FluidSolutionGradientTags(SolutionGradientTags):
    # DUDX = 1
    # DUDY = 2
    # DUDZ = 3
    # DVDX = 4
    # DVDY = 5
    # DVDZ = 6
    # DWDX = 7
    # DWDY = 8
    # DWDZ = 9
    GRADVEL = 10
    VORTICITY = 11
    # DTDX = 12
    # DTDY = 13
    # DTDZ = 14
    GRADT = 15
    SHEARSTRESS = 16
    HEATFLUX = 17
    DIVERGENCEVEL = 18
    VISCOUSDISSIPATION = 19
    HEATFLUX_MINUS_VISCOUSDISSIPATION = 20
    SHEARSTRESS_XX = 21
    SHEARSTRESS_XY = 22
    SHEARSTRESS_YY = 23
    HEATFLUX_X = 24
    HEATFLUX_Y = 25
    DIFFUSIVEFLUX_X1 = 26
    DIFFUSIVEFLUX_X2 = 27
    DIFFUSIVEFLUX_X3 = 28
    DIFFUSIVEFLUX_X4 = 29
    DIFFUSIVEFLUX_Y1 = 30
    DIFFUSIVEFLUX_Y2 = 31
    DIFFUSIVEFLUX_Y3 = 32
    DIFFUSIVEFLUX_Y4 = 33

#* --- TAGS FOR COORDINATES SYSTEMS TYPES --- *#
class CoordinatesSystemType(Enum):
    CARTESIAN = 1
    CYLINDRICAL = 2

#* --- TAGS FOR PROJECTION TYPES --- *#
class ProjectionType(Enum):
    NOPROJECTION = 1
    NORMAL = 2
    TANGENT = 3
    NORMALNORMAL = 4
    NORMALTANGENT = 5

#* --- TAGS FOR OUTPUT FILE TYPE --- *#
class OutputFileType(Enum):
    TEXT = 1
    CSV = 2
