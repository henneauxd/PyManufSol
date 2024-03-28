from PhysicalModels.GeneralPhysicalModels import GeneralPhysicalModels
from BoundaryConditions.GeneralBoundaryConditions import GeneralBoundaryConditions
from BoundaryConditions.BoundaryGeometry import *
from ProblemSolving.GeneralParametersSolver import GeneralParametersSolver
from Common.Utilities import *
from Common.MMSTags import VarSetTags, DefaultVarSetTags, SolutionTags, SolutionGradientTags, ProjectionType
from Outputs.PlotOverArea import PlotOver2DArea
from typing import Union
import sympy as sp
import copy

class QuantityInfoForPlot:

    def __init__(self, tag: Union[SolutionTags, SolutionGradientTags], isDeriv: bool = False, projectToDir: ProjectionType = ProjectionType.NOPROJECTION, evalAtTime: Optional[float] = None, plotVecArrows: bool = False):
        self.tag = tag
        self.isDeriv = isDeriv
        self.projectToDir = projectToDir
        self.evalAtTime = evalAtTime
        self.plotVecArrows = plotVecArrows

    def getQuantityTag(self):
        return self.tag

    def isDerivQuantity(self):
        return self.isDeriv

    def getProjectionType(self):
        return self.projectToDir

    def getEvalAtTime(self):
        return self.evalAtTime
    
    def isVectorArrowsPlot(self):
        return self.plotVecArrows

class GeneralProblemHandler:

    def __init__(self, physical_models: list[GeneralPhysicalModels], conditions: Optional[list[GeneralBoundaryConditions]] = None, solver_manual = True, solver_implicit = True):
        self.physical_models = dict()
        for mod in physical_models:
            tag_mod = mod.getPhysicalModelTag()
            if tag_mod in self.physical_models:
                raise ValueError("Different physical models have the same tag.")
            else:
                self.physical_models[tag_mod] = mod
        self.conditions = dict()
        self.params_sol = dict()
        if isinstance(conditions, list):
            for cond in conditions:
                tag_cond = cond.getBoundaryGeometry().getBoundaryTag()
                if tag_cond in self.conditions:
                    raise ValueError("Different boundary conditions have the same tag.")
                else:
                    self.conditions[tag_cond] = cond
            if any(c.getNumberImposedConditions() > 0 for c in conditions): 
                params_solver = GeneralParametersSolver(physical_models)
                self.params_sol = params_solver.solveForManufSolParameters(conditions, True, False, None, solver_manual, solver_implicit)
    
    def getParametricSolCoeff(self) -> dict:
        return self.params_sol

    def getThisPhysicalModel(self, tag: str) -> GeneralPhysicalModels:
        if tag not in self.physical_models:
            raise ValueError("Could not find physical model with tag %s."%tag)
        return self.physical_models[tag]

    def getThisBoundary(self, tag: str) -> GeneralBoundaryConditions:
        if tag not in self.conditions:
            raise ValueError("Could not find boundary with tag %s."%tag)
        return self.conditions[tag]

    def copyAndChangeCoordinatesSystem(self, new_spatial_sym_var: list, new_coord_system: CoordinatesSystemType, translation: Optional[list[float]] = None, rotation: Optional[list[list[int,float]]] = None):
        new_problem = copy.deepcopy(self)
        first_physical_model = list(new_problem.physical_models.values())[0]
        current_sym_var = first_physical_model.getSpatialSymVar()
        current_coord_system = first_physical_model.getCoordSystem()

        for key in new_problem.params_sol:
            new_problem.params_sol[key] = expressScalarInNewCoordSystem(new_problem.params_sol[key], current_sym_var, current_coord_system, new_spatial_sym_var, new_coord_system)

        for key in new_problem.conditions:
            new_problem.conditions[key].changeManufSolCoordSystem(new_spatial_sym_var, new_coord_system, translation, rotation)

        for key in new_problem.physical_models:
            new_problem.physical_models[key].changeManufSolCoordSystem(new_spatial_sym_var, new_coord_system, translation, rotation)

        return new_problem

    # Printing
    def printMMSSourceTermInFile(self, filePath: str = "", fileType: OutputFileType = OutputFileType.TEXT, simplify = False, printOnScreen: bool = False) -> None:
        for key in self.physical_models:
            print("")
            print("===== PRINTING MMS SOURCE TERM FOR %s PHYSICAL MODEL ====="%str(key))
            src = self.physical_models[key].computeMMSSourceTerm(self.getParametricSolCoeff())
            file_prefix = filePath if filePath != "" else defaultFilePath()
            file_name = file_prefix + "MMS_Source_" + str(key) + extensionName(fileType)
            file = openFile(file_name, fileType)
            for i in range(len(src)):
                if simplify and isinstance(src[i], sp.Expr):
                    src[i] = sp.simplify(src[i])
                s_str = str(src[i]).replace("**","^").replace(" ","")
                writeOneLineInFile(file, s_str, fileType)
                if printOnScreen:
                    print(" ")
                    print("Source(%d):  %s"%(i, s_str))
            closeFile(file)

    def printSolutionVectorInFile(self, list_tags: list[SolutionTags] = [], var_set: Optional[VarSetTags] = DefaultVarSetTags, filePath: str = "", fileType: OutputFileType = OutputFileType.TEXT, simplify = False, printOnScreen: bool = False) -> None:
        for key in self.physical_models:
            print("")
            print("===== PRINTING SOLUTION VECTOR FOR %s PHYSICAL MODEL ====="%str(key))
            sol_vec = self.physical_models[key].getSolVectorFromTags(list_tags, var_set, self.getParametricSolCoeff())
            file_prefix = filePath if filePath != "" else defaultFilePath()
            file_name = file_prefix + "MMS_Sol_" + str(key) + extensionName(fileType)
            file = openFile(file_name, fileType)
            for i in range(len(sol_vec)):
                if simplify and isinstance(sol_vec[i], sp.Expr):
                    sol_vec[i] = sp.simplify(sol_vec[i])
                s_str = str(sol_vec[i]).replace("**","^").replace(" ","")
                writeOneLineInFile(file, s_str, fileType)
                if printOnScreen:
                    print("Solution(%d):  %s"%(i, s_str))
            closeFile(file)

    def printAnyFieldDataInFile(self, list_tags: list, filePath: str = "", fileType: OutputFileType = OutputFileType.TEXT, simplify = False, printOnScreen: bool = False) -> None:
        
        num_params = self.getParametricSolCoeff()
        for key in self.physical_models:
            print("")
            print("===== PRINTING VARIABLES FOR %s PHYSICAL MODEL ====="%str(key))
            file_prefix = filePath if filePath != "" else defaultFilePath()
            file_name = file_prefix + "MMS_Var_" + str(key) + extensionName(fileType)
            file = openFile(file_name, fileType)
            for tag in list_tags:
                sol_vec = 0.0
                if isinstance(tag, SolutionTags):
                    sol_vec = self.physical_models[key].getSolTag(tag, num_params)
                elif isinstance(tag, SolutionGradientTags):
                    sol_vec = self.physical_models[key].getGradVarTag(tag, num_params, True)

                if not isinstance(sol_vec, list):
                    sol_vec = [sol_vec]
                for i in range(len(sol_vec)):
                    if simplify and isinstance(sol_vec[i], sp.Expr):
                        sol_vec[i] = sp.simplify(sol_vec[i])
                    s_str = str(sol_vec[i]).replace("**","^").replace(" ","")
                    writeOneLineInFile(file, s_str, fileType)
                    if printOnScreen:
                        print("Var(%d):  %s"%(i, s_str))
            closeFile(file)

    def evaluateJumpAtInterface(self, quantities: list[QuantityInfoForPlot], sideInterfaces: dict(), iso0_geo: GeneralBoundaryGeometry, fixed_coord: list[sp.Expr], val_fixed_coord: list[float], time_coord: sp.Expr, val_time_coord: float):
        num_params = self.getParametricSolCoeff()
        first_physical_model = list(self.physical_models.values())[0]
        sym_var = first_physical_model.getSpatialSymVar()
        iso0_eq = iso0_geo.getBoundaryEquation()
        iso0_eq_subs = iso0_eq
        iso0_eq_subs = iso0_eq_subs.subs({time_coord:val_time_coord})
        spatial_coord = dict()
        spatial_coord_val = []
        for i,j in zip(fixed_coord,val_fixed_coord):
            iso0_eq_subs = iso0_eq_subs.subs({i:j})
            spatial_coord[i] = j
            spatial_coord_val.append(j)
        freeCoord = [i for i in sym_var if i not in fixed_coord]
        freeCoord = freeCoord[0]
        freeCoordValue = sp.solve(iso0_eq_subs, freeCoord)
        spatial_coord[freeCoord] = freeCoordValue[0]
        spatial_coord_val.append(freeCoordValue[0])
        
        if len(freeCoordValue) != 0:
            val = iso0_eq.subs(freeCoord, freeCoordValue[0])
        else:
            val = iso0_eq
        if isinstance(val_fixed_coord, list):
            for j in range(len(fixed_coord)):
                val = val.subs(fixed_coord[j], val_fixed_coord[j])

        for q in quantities:
            tag = q.getQuantityTag()
            name_plot = str(tag.name)
            proj = q.getProjectionType()
            if proj != ProjectionType.NOPROJECTION:
                name_plot = name_plot + " " + str(proj.name)
            jump = 0
            for side, side_value in sideInterfaces.items():
                sol = 0.0
                if isinstance(tag, SolutionTags):
                    if not q.isDerivQuantity():
                        sol = self.physical_models[side].getSolTag(tag, num_params)
                    else:
                        sol = self.physical_models[side].getGradSolTag(tag, sym_var, num_params, True)
                elif isinstance(tag, SolutionGradientTags):
                    sol = self.physical_models[side].getGradVarTag(tag, num_params, True)
                if isinstance(q.getEvalAtTime(), float) and len(self.physical_models[side].getTemporalSymVar()) > 0:
                    sol = subsNumParamsBis(sol, [self.physical_models[side].getTemporalSymVar()[0]], [q.getEvalAtTime()])
                
                sol_proj = projectBoundaryVarAlongDir(sol, iso0_geo, proj)
                sol_proj = sol_proj.subs(spatial_coord)
                jump -= sol_proj*side_value

            print(">>> Jump on %s at (%2.3f,%2.3f) on iso0 at time %2.3f: %2.12f"%(name_plot,spatial_coord_val[0],spatial_coord_val[1],val_time_coord,jump))

    # Plotting
    def plotQuantitiesAlongOneDimLinesOverThisArea(self, quantities: list[QuantityInfoForPlot], line_plot: list[GeneralBoundaryGeometry], area: PlotOver2DArea, sideInterfaces: dict(), nb_pts: int, sampleDir: int):
        line_type = ['-', '--', '-.', ':']
        line_color = plt.get_cmap("tab10")
        
        num_params = self.getParametricSolCoeff()
        first_physical_model = list(self.physical_models.values())[0]
        sym_var = first_physical_model.getSpatialSymVar()
        for q in quantities:
            tag = q.getQuantityTag()
            name_plot = str(tag.name)
            proj = q.getProjectionType()
            if proj != ProjectionType.NOPROJECTION:
                name_plot = name_plot + " " + str(proj.name)
            plt.figure()
            ind_1 = 0
            for side in sideInterfaces:
                name_plot_phase = name_plot + ", " + str(side)
                sol = 0.0
                if isinstance(tag, SolutionTags):
                    if not q.isDerivQuantity():
                        sol = self.physical_models[side].getSolTag(tag, num_params)
                    else:
                        sol = self.physical_models[side].getGradSolTag(tag, sym_var, num_params, True)
                elif isinstance(tag, SolutionGradientTags):
                    sol = self.physical_models[side].getGradVarTag(tag, num_params, True)
                if isinstance(q.getEvalAtTime(), float) and len(self.physical_models[side].getTemporalSymVar()) > 0:
                    # local_dic = dict()
                    # local_dic[self.physical_models[side].getTemporalSymVar()[0]] = q.getEvalAtTime()
                    # sol = subsNumParams(sol, local_dic)
                    sol = subsNumParamsBis(sol, [self.physical_models[side].getTemporalSymVar()[0]], [q.getEvalAtTime()])
                    # sol = sol.subs({self.physical_models[side].getTemporalSymVar()[0]: q.getEvalAtTime()})
                interface = area.getInterfaces()
                ind_2 = 0
                for line in line_plot:
                    if len(interface) > 0:
                        sol_proj = projectBoundaryVarAlongDir(sol, interface[0], proj)
                    else:
                        sol_proj = projectBoundaryVarAlongDir(sol, line, proj)
                    area.plotThisFieldAlongThoseStraightLines(sol_proj, [line.getBoundaryEquation()], nb_pts, sideInterfaces[side], sampleDir, name_plot_phase, line_color(ind_2), line_type[ind_1])
                    ind_2 = ind_2 + 1
                ind_1 = ind_1 + 1
            plt.xlabel(str(sym_var[sampleDir]), size = 26)
            plt.ylabel(name_plot, size = 26)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            ax = plt.gca()
            figure = plt.gcf()
            figure.set_size_inches(9*1.6, 9)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2)
            # plt.legend()
            # plt.legend(prop={'size': 10}, handlelength=3.5)
            format_save='png'
            # plt.savefig(name_plot+'.'+format_save, format=format_save, dpi = 500)
            plt.show()

    def plotQuantitiesOverThisTwoDimArea(self, quantities: list[QuantityInfoForPlot], area: PlotOver2DArea, sideInterfaces: dict(), nb_pts: int):
        num_params = self.getParametricSolCoeff()
        first_physical_model = list(self.physical_models.values())[0]
        sym_var = first_physical_model.getSpatialSymVar()
        for q in quantities:
            if q.isVectorArrowsPlot():
                fig, axes = plt.subplots(figsize =(8, 8))
            else:
                fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

            tag = q.getQuantityTag()
            name_plot = str(tag.name)
            proj = q.getProjectionType()
            if proj != ProjectionType.NOPROJECTION:
                name_plot = name_plot + " " + str(proj.name)
            for side in sideInterfaces:
                name_plot_phase = name_plot + ", " + str(side)
                sol = 0.0
                sol = 0.0
                if isinstance(tag, SolutionTags):
                    if not q.isDerivQuantity():
                        sol = self.physical_models[side].getSolTag(tag, num_params)
                    else:
                        sol = self.physical_models[side].getGradSolTag(tag, sym_var, num_params, True)
                elif isinstance(tag, SolutionGradientTags):
                    sol = self.physical_models[side].getGradVarTag(tag, num_params, True)
                if isinstance(q.getEvalAtTime(), float) and len(self.physical_models[side].getTemporalSymVar()) > 0:
                    sol = subsNumParamsBis(sol, [self.physical_models[side].getTemporalSymVar()[0]], [q.getEvalAtTime()])
                interface = area.getInterfaces()
                if len(interface) > 0:
                    sol = projectBoundaryVarAlongDir(sol, interface[0], proj)

                if isinstance(sol,list) and q.isVectorArrowsPlot():
                    area.plotThisVectorOver2DArea(sol, nb_pts, sideInterfaces[side], axes, name_plot_phase)
                else:
                    area.plotThisFieldOver2DArea(sol, nb_pts, sideInterfaces[side], axes, name_plot_phase)  
            axes.legend()
            axes.legend(prop={'size': 16}, handlelength=3.5)
            axes.set_xlabel(str(sym_var[0]), size = 26)
            axes.set_ylabel(str(sym_var[1]), size = 26)
            # axes.set_zlabel('', size = 26)
            # axes.set_zlim(0.5,1.35)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            axes.tick_params(labelsize=16)

            figure = plt.gcf()
            # figure.set_size_inches(9*1.6, 9)
            figure.set_size_inches(9, 9)
            format_save='png'
            # axes.view_init(elev=32., azim=-165)
            # axes.view_init(elev=27., azim=-155)
            # axes.view_init(elev=27., azim=-135)
            # plt.savefig(name_plot+'.'+format_save, format=format_save, dpi = 300)
            plt.show()

            # axes.set_zlabel(name_plot)
            plt.show()

