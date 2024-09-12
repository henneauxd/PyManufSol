from PhysicalModels.GeneralPhysicalModels import GeneralPhysicalModels
from BoundaryConditions.BoundaryGeometry import *
from Common.MMSTags import SolutionTags, SolutionGradientTags, ProjectionType
from Common.Utilities import *
from typing import Union


class GeneralBoundaryConditions:
    def __init__(self, physical_models: list[GeneralPhysicalModels], boundary_geometry: GeneralBoundaryGeometry):
        self.physical_models = physical_models
        self.boundary_geometry = boundary_geometry
        self.nb_phases = len(physical_models)
        if self.nb_phases > 2 or self.nb_phases < 1:
            raise ValueError("The number of physical models must be equal to one or two.")
        sym_var = self.physical_models[0].getSymVar()
        if self.nb_phases > 1:
            for i in range(1,self.nb_phases):
                if not all(j==k for j,k in zip(self.physical_models[i].getSymVar(), sym_var)):
                    raise ValueError("The independent spatial and temporal coordinates must be identical between the different physical models provided.")
        self.conditions = []
        self.conditions_with_derivs = 0
        self.domain_dim = self.physical_models[0].getDomainDim()

    def getImposedConditions(self) -> list[sp.Expr]:
        return self.conditions

    def getNumberImposedConditions(self) -> int:
        return len(self.conditions)

    def getNumberConditionsInvolvingSolDerivs(self) -> int:
        return self.conditions_with_derivs

    def getBoundaryGeometry(self) -> GeneralBoundaryGeometry:
        return self.boundary_geometry

    def checkCondition(self, cond: sp.Expr, sym_var, isDirichletType: bool):
        if isDirichletType:
            pass
        elif any(var in cond.free_symbols for var in sym_var):
            raise ValueError("This non-Dirichlet boundary condition depends on spatial variables. This means that the unknowns manuf sol parameters will be dependent on the spatila coordinates and that a DAE system would need to be solved to find their expressions since a non-Dirichilet BC type has been prescribed. However, this is not possible for the moment")
        pass

    # def projectBoundaryVarAlongDir(self, var, boundary_geo: GeneralBoundaryGeometry, projectToDir: ProjectionType):
    #     if not isinstance(var, list) or projectToDir == ProjectionType.NOPROJECTION:
    #         return var
    #     else:
    #         n_boundary = boundary_geo.getNormalToBoundary()
    #         t_boundary = boundary_geo.getTangentsToBoundary()
    #         size_var = len(var)
    #         if projectToDir == ProjectionType.NORMAL:
    #             if size_var != len(n_boundary):
    #                 raise ValueError("Size of vector to be projected different from size of normal vector.")
    #             return projectThisVecOrTensorAlongThisDirection(var, n_boundary)
    #         elif projectToDir == ProjectionType.TANGENT:
    #             if size_var != len(t_boundary[0]):
    #                 raise ValueError("Size of vector to be projected different from size of tangent vector.")
    #             if boundary_geo.getDomainDim() == 2:
    #                 return projectThisVecOrTensorAlongThisDirection(var, t_boundary[0])
    #             elif boundary_geo.getDomainDim() == 3:
    #                 return [projectThisVecOrTensorAlongThisDirection(var, t_boundary[0]), projectThisVecOrTensorAlongThisDirection(var, t_boundary[1])]
    #         elif projectToDir == ProjectionType.NORMALNORMAL:
    #             return projectThisTensorAlongThoseDirections(var, n_boundary, n_boundary)
    #         elif projectToDir == ProjectionType.NORMALTANGENT: 
    #             if boundary_geo.getDomainDim()== 2:
    #                 return projectThisTensorAlongThoseDirections(var, n_boundary, t_boundary[0])
    #             elif boundary_geo.getDomainDim() == 3:
    #                 return [projectThisTensorAlongThoseDirections(var, n_boundary, t_boundary[0]), projectThisTensorAlongThoseDirections(var, n_boundary, t_boundary[1])]
    #         else:
    #             raise ValueError("Unknown projection type on boundaries.")

    def produceBC(self, boundary_geo: GeneralBoundaryGeometry, tag: Union[SolutionTags, SolutionGradientTags], isDirichletType: bool, imposed_val: list, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False) -> list:
        var = []
        conditions = []
        for i in range(self.nb_phases):
            sol = 0.0
            if isinstance(tag, SolutionTags):
                if isDirichletType:
                    sol = self.physical_models[i].getSolTag(tag, None)
                else:
                    sol = self.physical_models[i].getGradSolTag(tag, boundary_geo.getSymVariables(), None, False)
            else:
                sol = self.physical_models[i].getGradVarTag(tag, None, False)
            var.append(projectBoundaryVarAlongDir(sol, boundary_geo, projectToDir))
            var[i] = var[i] if type(var[i]) is list else [var[i]] 
        if project_imposed_val:
            imposed_val = projectBoundaryVarAlongDir(imposed_val, boundary_geo, projectToDir)
            imposed_val = imposed_val if type(imposed_val) is list else [imposed_val]
        if len(var[0]) != len(imposed_val):
            raise ValueError("Size of variables to be imposed different from size of Dirichlet imposed values.")
        for i in range(len(var[0])):
            if self.nb_phases == 2:
                conditions.append(var[0][i]-var[1][i]-imposed_val[i])
            else:
                conditions.append(var[0][i]-imposed_val[i])
        return conditions
    
    def addBC(self, tag: Union[SolutionTags, SolutionGradientTags], isDirichletType: bool, imposed_val: list, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        for i in self.produceBC(self.boundary_geometry, tag, isDirichletType, imposed_val, projectToDir, project_imposed_val):
            self.conditions.append(i)

    def addBCAndSubsituteBoundaryEq(self, tag: Union[SolutionTags, SolutionGradientTags], isDirichletType: bool, imposed_val: list, coord_to_be_substituted: sp.Expr, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        sym_var = self.boundary_geometry.getSymVariables()
        if coord_to_be_substituted not in sym_var:
            raise ValueError("The coordinate to be replaced from the boundary eq does not exist in the boundary equation.")
        coord_to_be_substituted_value = sp.solve(self.boundary_geometry.getBoundaryEquation(), coord_to_be_substituted)
        for i in self.produceBC(self.boundary_geometry, tag, isDirichletType, imposed_val, projectToDir, project_imposed_val):
            if len(coord_to_be_substituted_value) != 0:
                val = i.subs(coord_to_be_substituted, coord_to_be_substituted_value[0])
            else:
                val = i
            self.checkCondition(val, sym_var, isDirichletType)
            self.conditions.append(val)

    def addBCAndSubsituteBoundaryPoint(self, tag: Union[SolutionTags, SolutionGradientTags], isDirichletType: bool, imposed_val: list, fixed_coord: list[sp.Expr], val_fixed_coord: Optional[list[float]] = None, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        if len(fixed_coord) != self.domain_dim-1:
            raise ValueError("Number of fixed coordinates incompatible with domain dimension.")
        sym_var = self.boundary_geometry.getSymVariables()
        freeCoord = [i for i in sym_var if i not in fixed_coord]
        freeCoord = freeCoord[0]
        freeCoordValue = sp.solve(self.boundary_geometry.getBoundaryEquation(), freeCoord)
        for i in self.produceBC(self.boundary_geometry, tag, isDirichletType, imposed_val, projectToDir, project_imposed_val):
            if len(freeCoordValue) != 0:
                val = i.subs(freeCoord, freeCoordValue[0])
            else:
                val = i
            if isinstance(val_fixed_coord, list):
                for j in range(len(fixed_coord)):
                    val = val.subs(fixed_coord[j], val_fixed_coord[j])
            self.checkCondition(val, sym_var, isDirichletType)
            self.conditions.append(val)

    def addDirichletBC(self, tag: SolutionTags, imposed_val: list, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        self.addBC(tag, True, imposed_val, projectToDir, project_imposed_val)

    def addDirichletBCAndSubsituteBoundaryEq(self, tag: SolutionTags, imposed_val: list, coord_to_be_substituted: sp.Expr, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        self.addBCAndSubsituteBoundaryEq(tag, True, imposed_val, coord_to_be_substituted, projectToDir, project_imposed_val)

    def addDirichletBCAndSubsituteBoundaryPoint(self, tag: SolutionTags, imposed_val: list, fixed_coord: list[sp.Expr], val_fixed_coord: Optional[list[float]] = None, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        self.addBCAndSubsituteBoundaryPoint(tag, True, imposed_val, fixed_coord, val_fixed_coord, projectToDir, project_imposed_val)

    def addNeumannBC(self, tag: Union[SolutionTags, SolutionGradientTags], imposed_val: list, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        self.addBC(tag, False, imposed_val, projectToDir, project_imposed_val)
        self.conditions_with_derivs = self.conditions_with_derivs + 1

    def addNeumannBCAndSubsituteBoundaryEq(self, tag: Union[SolutionTags, SolutionGradientTags], imposed_val: list, coord_to_be_substituted: sp.Expr, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        self.addBCAndSubsituteBoundaryEq(tag, False, imposed_val, coord_to_be_substituted, projectToDir, project_imposed_val)
        self.conditions_with_derivs = self.conditions_with_derivs + 1

    def addNeumannBCAndSubsituteBoundaryPoint(self, tag: Union[SolutionTags, SolutionGradientTags], imposed_val: list, fixed_coord: list[sp.Expr], val_fixed_coord: Optional[list[float]] = None, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
        self.addBCAndSubsituteBoundaryPoint(tag, False, imposed_val, fixed_coord, val_fixed_coord, projectToDir, project_imposed_val)
        self.conditions_with_derivs = self.conditions_with_derivs + 1

    def changeManufSolCoordSystem(self, new_spatial_sym_var: list, new_coord_system: CoordinatesSystemType, translation: Optional[list[float]] = None, rotation: Optional[list[list[int,float]]] = None):
        current_coord_system = self.physical_models[0].getCoordSystem()

        for i in range(len(self.conditions)):
            current_spatial_sym_var = self.boundary_geometry.getSymVariables()
            new_cond = self.conditions[i]
            if current_coord_system != new_coord_system:
                new_cond = expressScalarInNewCoordSystem(new_cond, current_spatial_sym_var, current_coord_system, new_spatial_sym_var, new_coord_system)
                current_spatial_sym_var = new_spatial_sym_var
            if isinstance(translation, list):
                new_cond = translateScalarInNewCoordSystem(new_cond, current_spatial_sym_var, new_spatial_sym_var, translation)
                current_spatial_sym_var = new_spatial_sym_var
            if isinstance(rotation, list):
                new_cond = rotateScalarInNewCoordSystem(new_cond, current_spatial_sym_var, new_spatial_sym_var, rotation)
                current_spatial_sym_var = new_spatial_sym_var
            self.conditions[i] = new_cond

        current_spatial_sym_var = self.boundary_geometry.getSymVariables()
        new_bnd_eq = self.boundary_geometry.getBoundaryEquation()
        if current_coord_system != new_coord_system:
            new_bnd_eq = expressScalarInNewCoordSystem(new_bnd_eq, current_spatial_sym_var, current_coord_system, new_spatial_sym_var, new_coord_system)
            current_spatial_sym_var = new_spatial_sym_var
        if isinstance(translation, list):
            new_bnd_eq = translateScalarInNewCoordSystem(new_bnd_eq, current_spatial_sym_var, new_spatial_sym_var, translation)
            current_spatial_sym_var = new_spatial_sym_var
        if isinstance(rotation, list):
            new_bnd_eq = rotateScalarInNewCoordSystem(new_bnd_eq, current_spatial_sym_var, new_spatial_sym_var, rotation)
            current_spatial_sym_var = new_spatial_sym_var
        self.boundary_geometry.setBoundaryEquation(new_bnd_eq)
        self.boundary_geometry.setSymVar(new_spatial_sym_var)


    # def addDirichletBCAndSubsituteBoundary(self, boundary_geo: GeneralBoundaryGeometry, tag: SolutionTags, imposed_val: list, indval_fixed_coord: list[list[int,float]], projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
    #     if len(indval_fixed_coord) != self.domain_dim-1:
    #         raise ValueError("Number of fixed coordinates incompatible with domain dimensions.")
    #     sym_var = boundary_geo.getSymVariables()
    #     imposedCoordInd = [indval_fixed_coord[i][0] for i in range(len(indval_fixed_coord))]
    #     freeCoordInd = [i for i in range(self.domain_dim) if i not in imposedCoordInd]
    #     if self.domain_dim == 2:
    #         freeCoordValue = sp.solve(boundary_geo.getBoundaryEquation().subs(sym_var[indval_fixed_coord[0][0]],indval_fixed_coord[0][1]), freeCoordInd)
    #     elif self.domain_dim == 3:
    #         freeCoordValue = sp.solve(boundary_geo.getBoundaryEquation().subs(sym_var[indval_fixed_coord[0][0]],indval_fixed_coord[0][1]).subs(sym_var[indval_fixed_coord[1][0]],indval_fixed_coord[1][1]), freeCoordInd)
    #     freeCoordValue = freeCoordValue[0]
    #     for i in self.produceDirichletBC(boundary_geo, tag, imposed_val, projectToDir, project_imposed_val):
    #         val = i.subs()
    #         self.conditions.append(val.subs(freeCoordValue))

    

# class GeneralBoundaryConditions:
#     def __init__(self, boundary_geo: GeneralBoundaryGeometry):
#         # self.physical_model = physical_models
#         self.boundary_geo = boundary_geo
#         self.conditions = []
#         self.domain_dim = boundary_geo.getDomainDim()
#         # self.nb_phases = len(physical_models)
#         # self.unknowns_params = []
#         # for i in range(self.nb_phases):
#         #     self.unknowns_params.append(physical_models[i].getManufSolContainer().getSymUnkownsParams())

#     def projectBoundaryVarAlongDir(self, var, boundary_geo: GeneralBoundaryGeometry, projectToDir: ProjectionType):
#         if not isinstance(var, list) or projectToDir == ProjectionType.NOPROJECTION:
#             return var
#         else:
#             n_boundary = boundary_geo.getNormalToBoundary()
#             t_boundary = boundary_geo.getTangentsToBoundary()
#             size_var = len(var)
#             if projectToDir == ProjectionType.NORMAL:
#                 if size_var != len(n_boundary):
#                     raise ValueError("Size of vector to be projected different from size of normal vector.")
#                 return projectThisVecOrTensorAlongThisDirection(var, n_boundary)
#             elif projectToDir == ProjectionType.TANGENT:
#                 if size_var != len(t_boundary[0]):
#                     raise ValueError("Size of vector to be projected different from size of tangent vector.")
#                 if self.domain_dim == 2:
#                     return projectThisVecOrTensorAlongThisDirection(var, t_boundary[0])
#                 elif self.domain_dim == 3:
#                     return [projectThisVecOrTensorAlongThisDirection(var, t_boundary[0]), projectThisVecOrTensorAlongThisDirection(var, t_boundary[1])]
#             elif projectToDir == ProjectionType.NORMALNORMAL:
#                 return projectThisTensorAlongThoseDirections(var, n_boundary, n_boundary)
#             elif projectToDir == ProjectionType.NORMALTANGENT: 
#                 if self.domain_dim == 2:
#                     return projectThisTensorAlongThoseDirections(var, n_boundary, t_boundary[0])
#                 elif self.domain_dim == 3:
#                     return [projectThisTensorAlongThoseDirections(var, n_boundary, t_boundary[0]), projectThisTensorAlongThoseDirections(var, n_boundary, t_boundary[1])]
#             else:
#                 raise ValueError("Unknown projection type on boundaries.")

#     def addDirichletBC(self, boundary_geo: GeneralBoundaryGeometry, physical_model: GeneralPhysicalModels, tag: SolutionTags, imposed_val: list, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, project_imposed_val: bool = False):
#         var = self.projectBoundaryVarAlongDir(physical_model.getSolTag(tag, None), boundary_geo, projectToDir)
#         var = var if type(var) is list else [var] 
#         if project_imposed_val:
#             imposed_val = self.projectBoundaryVarAlongDir(imposed_val, boundary_geo, projectToDir)
#             imposed_val = imposed_val if type(imposed_val) is list else [imposed_val]
#         if len(var) != len(imposed_val):
#             raise ValueError("Size of variables to be imposed different from size of Dirichlet imposed values.")
#         for i in range(len(var)):
#             self.conditions.append(var[i]-imposed_val[i])

    
        



