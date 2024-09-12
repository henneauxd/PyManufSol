
from PhysicalModels.GeneralPhysicalModels import GeneralPhysicalModels
from BoundaryConditions.GeneralBoundaryConditions import GeneralBoundaryConditions
import sympy as sp
from sympy.solvers.solveset import linsolve
from sympy.solvers import solve, nsolve
from Common.Utilities import is_linear

class GeneralParametersSolver:

    # def __init__(self, physical_models: list[GeneralPhysicalModels], conditions: list[GeneralBoundaryConditions]):
    #     self.physical_models = physical_models
    #     self.conditions = conditions
    #     self.nb_phases = len(physical_models)
    #     self.unknowns_params = []
    #     sym_var = self.physical_models[0].getSymVar()
    #     for i in range(1,self.nb_phases):
    #         if not all(j==k for j,k in zip(self.physical_models[i].getSymVar(),sym_var)):
    #             raise ValueError("The independent spatial and temporal coordinates must be identical between the different physical models provided.")
    #     for i in range(self.nb_phases):
    #         self.unknowns_params.append(physical_models[i].getManufSolContainer().getSymUnkownsParams())
    #     self.nb_conditions = 0 if not isinstance(conditions, list) else len(conditions)
    #     if self.nb_conditions != len(self.unknowns_params):
    #         raise ValueError("The number of conditions on the manufactured solution is not equal to the number of unknwons parameters to be determined.")
    #     if self.needDAESystemSolve():
    #         raise ValueError("To find the manuf sol parameters, a DAE system would need to be solved because the set conditions involve conditions with derivatives and that some manuf sol parameters have been identified to be dependent on spatial coordinates. This is not possible for the moment.")

    def __init__(self, physical_models: list[GeneralPhysicalModels]):
        nb_phases = len(physical_models)
        self.unknowns_params = []
        self.sym_var = physical_models[0].getSymVar()
        for i in range(1,nb_phases):
            if not all(j==k for j,k in zip(physical_models[i].getSymVar(),self.sym_var)):
                raise ValueError("The independent spatial and temporal coordinates must be identical between the different physical models provided.")
        for i in range(nb_phases):
            self.unknowns_params.extend(physical_models[i].getManufSolContainer().getSymUnkownsParams())
        self.spatial_sym_var = physical_models[0].getSpatialSymVar()

    def needDAESystemSolve(self, conditions: list[GeneralBoundaryConditions]) -> bool:
        needDAE = False
        for i in conditions:
            if i.getNumberConditionsInvolvingSolDerivs() > 0:
                for j in i.getImposedConditions():
                    if any(var in j.free_symbols for var in self.spatial_sym_var):
                        needDAE = True
        return needDAE

    def solveForManufSolParametersFromGivenConditions(self, conditions: list[sp.Expr], unknown_params: list[sp.Expr], checkIfLinear: bool = True, mySimplify: bool = False, myRational = None, myManual: bool = True, myImplicit: bool = True):
        if len(conditions) != len(unknown_params):
            raise ValueError("The number of conditions on the manufactured solution (%d) is not equal to the number of unknwons parameters to be determined (%d)."%(len(conditions), len(unknown_params)))
        eval_param_sol = dict()
        bool_dict = False
        if checkIfLinear and all(is_linear(i, unknown_params) for i in conditions):
            param_sol = linsolve(conditions, tuple(unknown_params))
            param_sol_final = param_sol.args[0]
        else:
            # x_0 =  [-202.267776549951, -202.267776549951, -236.507447722715, 1185.70815302973, -1125.26576706481, -465.092344600401, 2422.26911011591, -2347.00853098756, 4197.85903310401, -12165.1201722485, 12951.5747070123]
            # param_sol_final = nsolve(conditions, tuple(unknown_params), x_0)
            param_sol_final = solve(conditions, tuple(unknown_params), simplify = mySimplify, rational = myRational, warn = True, manual = myManual, implicit = myImplicit)
            bool_dict = type(param_sol_final) is dict
            if bool_dict == False:
                param_sol_final = param_sol_final[0]
            else:
                eval_param_sol = param_sol_final
        if bool_dict == False:
            for i in range(len(param_sol_final)):
                eval_param_sol[unknown_params[i]] = param_sol_final[i]
        return eval_param_sol

    def solveForManufSolParameters(self, conditions: list[GeneralBoundaryConditions], checkIfLinear: bool = True, mySimplify: bool = False, myRational = None, myManual: bool = True, myImplicit: bool = True):
        if self.needDAESystemSolve(conditions):
            raise ValueError("To find the manuf sol parameters, a DAE system would need to be solved because the set conditions involve conditions with derivatives and that some manuf sol parameters have been identified to be dependent on spatial coordinates. This is not possible for the moment.")
        list_cond = []
        for i in conditions:
            list_cond.extend(i.getImposedConditions())
        return self.solveForManufSolParametersFromGivenConditions(list_cond, self.unknowns_params, checkIfLinear, mySimplify, myRational, myManual, myImplicit)
        

        
        


    
