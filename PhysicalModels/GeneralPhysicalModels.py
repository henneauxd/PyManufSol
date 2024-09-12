import sympy as sp
from typing import Optional
from ManufSolution.ManufSolContainer.GeneralManufSolContainer import GeneralManufSolContainerSinglePhase
from Common.MMSTags import MMSSourceTermTags, SolutionTags, SolutionGradientTags, CoordinatesSystemType, VarSetTags, DefaultVarSetTags
from Common.Utilities import *
from typing import Union
class GeneralPhysicalModels:
    """ Abstract class defining the common functions required 
    to specify the bulk governing equations of the problem. 
    
    Attributes:
        manuf_sol_dict: dictionary containing manufactured solution
        domain_dim: spatial dimension of the problem
        indexTemporalIndepVar: determines if the problem is steady or not
    """
    def __init__(self, tag: str, manuf_sol_container: GeneralManufSolContainerSinglePhase, domain_dim: int, coord_system: CoordinatesSystemType = CoordinatesSystemType.CARTESIAN):
        self.tag = tag
        self.manuf_sol_container = manuf_sol_container
        self.spatial_sym_var = self.manuf_sol_container.getSpatialSymVar()
        self.temporal_sym_var = self.manuf_sol_container.getTemporalSymVar()
        self.isSteady = len(self.temporal_sym_var) == 0
        self.coord_system = coord_system
        self.domain_dim = domain_dim
        if self.domain_dim != manuf_sol_container.getDim():
            raise ValueError("Number of spatial dimensions of physical model different from provided solution vector")
        if self.domain_dim != len(self.spatial_sym_var):
            raise ValueError("Number of spatial dimensions different from number of spatial indices")

    # Getters
    def getPhysicalModelTag(self):
        return self.tag

    def getDomainDim(self):
        return self.domain_dim

    def getSymVar(self):
        allSymVar = list(self.getSpatialSymVar())
        if not self.isSteady:
            allSymVar.append(self.getTemporalSymVar()[0])
        return tuple(allSymVar)

    def getSpatialSymVar(self):
        return self.spatial_sym_var

    def getTemporalSymVar(self):
        return self.temporal_sym_var

    def getCoordSystem(self) -> CoordinatesSystemType:
        return self.coord_system

    def getManufSolContainer(self) -> GeneralManufSolContainerSinglePhase:
        return self.manuf_sol_container

    # Setters
    def changeManufSolCoordSystem(self, new_spatial_sym_var: list, new_coord_system: CoordinatesSystemType, translation: Optional[list[float]] = None, rotation: Optional[list[list[int,float]]] = None):
        self.manuf_sol_container.changeManufSolCoordSystem(self.getCoordSystem(), new_spatial_sym_var, new_coord_system, translation, rotation)
        self.coord_system = new_coord_system
        self.spatial_sym_var = new_spatial_sym_var

    def solVector(self, num_params: Optional[dict] = None) -> list:
        list_var = []
        for tag in self.manuf_sol_container.getDicManufSolPerTag():
            # list_var.append(self.manuf_sol_container.getDicManufSolPerTag()[tag])
            list_var.append(self.getSolTag(tag, num_params))
        return list_var
        # return subsNumParams(list_var, num_params)

    def gradSolVector(self,  num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        list_var = []
        for tag in self.manuf_sol_container.getDicManufSolPerTag():
            list_var.append(self.getGradSolTag(tag, self.spatial_sym_var, num_params, replace_params_before_deriv))
        return list_var
    
    def getMMSSourceTag(self, tag: MMSSourceTermTags, num_params: Optional[dict] = None):
        var = 0.0
        if not isinstance(tag, MMSSourceTermTags):
            raise ValueError("The tag should be of 'MMSSourceTermTags' type")
        
        if tag == MMSSourceTermTags.MMS_SOURCE_CONVECTIVE:
            var = self.computeMMSConvectiveSourceTerm(num_params)
        elif tag == MMSSourceTermTags.MMS_SOURCE_DIFFUSIVE:
            var = self.computeMMSDiffusiveSourceTerm(num_params)
        elif tag == MMSSourceTermTags.MMS_SOURCE_SRC:
            var = self.computeMMSSourceSourceTerm(num_params)
        elif tag == MMSSourceTermTags.MMS_SOURCE_UNSTEADY:
            var = self.computeMMSUnsteadySourceTerm(num_params)
        elif tag == MMSSourceTermTags.MMS_SOURCE_FULL:
            var = self.computeMMSSourceTerm(num_params)
        elif tag == MMSSourceTermTags.MMS_SOURCE_CONVECTIVE_OVER_DIFFUSIVE or tag == MMSSourceTermTags.MMS_SOURCE_DIFFUSIVE_OVER_CONVECTIVE:
            conv_lst = self.getMMSSourceTag(MMSSourceTermTags.MMS_SOURCE_CONVECTIVE,num_params)
            diff_lst = self.getMMSSourceTag(MMSSourceTermTags.MMS_SOURCE_DIFFUSIVE,num_params)
            if tag == MMSSourceTermTags.MMS_SOURCE_CONVECTIVE_OVER_DIFFUSIVE:
                var = [conv_lst[i]/diff_lst[i] for i in range(len(conv_lst))]
            else:
                var = [diff_lst[i]/conv_lst[i] for i in range(len(conv_lst))]
        else:
            raise ValueError("The tag for the MMS source term is unknown.")
        return subsNumParams(var, num_params)

    def getSolTag(self, tag: SolutionTags, num_params: Optional[dict] = None):
        if tag not in self.manuf_sol_container.getDicManufSolPerTag():
            raise ValueError("The solution tag is not associated to any key in the dict containing the symbolic solution.")
        var = self.manuf_sol_container.getDicManufSolPerTag()[tag]
        return subsNumParams(var, num_params)

    def getSolVectorFromTags(self, list_tags: list[SolutionTags] = [], var_set: Optional[VarSetTags] = DefaultVarSetTags, num_params: Optional[dict] = None) -> list:
        tags = list_tags if len(list_tags) > 0 else self.getManufSolContainer().getSolsTagsFromVarSetChoice(var_set)
        return [self.getSolTag(t, num_params) for t in tags]

    def getGradSolTag(self, tag: SolutionTags, indep_vars: list, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        var = self.getSolTag(tag, None)
        var = var if type(var) is list else [var]
        if replace_params_before_deriv:
            var = subsNumParams(var, num_params)
        nb_row = len(var)
        nb_col = len(indep_vars)
        var_derivs = initListOfLists(nb_row, nb_col) 
        if nb_row > 1:
            copyListOfLists(computeGradOfVector(var, indep_vars, self.coord_system), var_derivs)
        else:
            var_derivs[0] = computeGradOfScalar(var[0], indep_vars, self.coord_system)
        
        return subsNumParams(var_derivs, num_params)

    def getDivergenceSolTag(self, tag: SolutionTags, indep_vars: list, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        var = self.getSolTag(tag, None)
        var = var if type(var) is list else [var]
        if replace_params_before_deriv:
            var = subsNumParams(var, num_params)
        nb_row = len(var)
        nb_col = len(indep_vars)
        var_derivs = initListOfLists(1, 1) 
        if len(var) > 1:
            var_derivs[0][0] = computeDivergenceOfVector(var, indep_vars, self.coord_system)

        return subsNumParams(var_derivs, num_params)

    def getDerivSolTag(self, tag: SolutionTags, indep_vars: list[list], num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        # if tag not in self.manuf_sol_container.getDicManufSolPerTag():
        #     raise ValueError("The solution tag is not associated to any key in the dict containing the symbolic solution.")
        var = self.getSolTag(tag, None)
        var = var if type(var) is list else [var]
        nb_row = len(var)
        nb_col = len(indep_vars)
        var_derivs = initListOfLists(nb_row, nb_col)# sp.zeros(nb_row, nb_col)
        for i in range(nb_row):
            for j in range(nb_col):
                if replace_params_before_deriv:
                    var_derivs[i][j] = subsNumParams(var[i],num_params)
                else:
                    var_derivs[i][j] = var[i]
                for ind_var in indep_vars[j]:
                        var_derivs[i][j] = sp.diff(var_derivs[i][j],ind_var)
        return subsNumParams(var_derivs, num_params)

    # def getAnyQuantityFromTag(self, tags: list, indep_vars: list[list], num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        
    def getGradVarTag(self, tag: SolutionGradientTags, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        raise NotImplementedError()

    def convectiveFlux(self, num_params: Optional[dict] = None):
        raise NotImplementedError()

    def diffusiveFlux(self, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        raise NotImplementedError()

    def sourceTerm(self, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        raise NotImplementedError()
    
    def computeMMSConvectiveSourceTerm(self, num_params: Optional[dict] = None) -> list:
        F_c = self.convectiveFlux(num_params)
        MMS_source = [0] * self.manuf_sol_container.getNumberFields()
        for i_var in range(self.manuf_sol_container.getNumberFields()):
            MMS_source[i_var] = 0
            geom_coeff = 1.0
            for i_dim in range(self.domain_dim):
                if self.coord_system == CoordinatesSystemType.CYLINDRICAL and i_dim == 1:
                    geom_coeff = 1.0 / self.spatial_sym_var[0]
                MMS_source[i_var] = MMS_source[i_var] + geom_coeff * (sp.diff(F_c[i_var][i_dim],self.spatial_sym_var[i_dim]))
        return MMS_source
    
    def computeMMSDiffusiveSourceTerm(self, num_params: Optional[dict] = None) -> list:
        F_d = self.diffusiveFlux(num_params, True)
        MMS_source = [0] * self.manuf_sol_container.getNumberFields()
        for i_var in range(self.manuf_sol_container.getNumberFields()):
            MMS_source[i_var] = 0
            geom_coeff = 1.0
            for i_dim in range(self.domain_dim):
                if self.coord_system == CoordinatesSystemType.CYLINDRICAL and i_dim == 1:
                    geom_coeff = 1.0 / self.spatial_sym_var[0]
                MMS_source[i_var] = MMS_source[i_var] - geom_coeff * (sp.diff(F_d[i_var][i_dim],self.spatial_sym_var[i_dim]))
        return MMS_source
    
    def computeMMSUnsteadySourceTerm(self, num_params: Optional[dict] = None) -> list:
        U = self.solVector(num_params)
        MMS_source = [0] * self.manuf_sol_container.getNumberFields()
        for i_var in range(self.manuf_sol_container.getNumberFields()):
            MMS_source[i_var] = 0
            if self.isSteady == False:
                MMS_source[i_var] = sp.diff(U[i_var],self.temporal_sym_var[0])
        return MMS_source
    
    def computeMMSSourceSourceTerm(self, num_params: Optional[dict] = None) -> list:
        S = self.sourceTerm(num_params, True)
        MMS_source = [0] * self.manuf_sol_container.getNumberFields()
        for i_var in range(self.manuf_sol_container.getNumberFields()):
            MMS_source[i_var] = 0
            MMS_source[i_var] = -S[i_var][0]
        return MMS_source

    def computeMMSSourceTerm(self, num_params: Optional[dict] = None) -> list:
        # U = self.solVector(num_params)
        # F_c = self.convectiveFlux(num_params)
        # F_d = self.diffusiveFlux(num_params, True)
        # S = self.sourceTerm(num_params, True)
        # MMS_source = [0] * self.manuf_sol_container.getNumberFields()
        # for i_var in range(self.manuf_sol_container.getNumberFields()):
        #     MMS_source[i_var] = 0
        #     geom_coeff = 1.0
        #     for i_dim in range(self.domain_dim):
        #         if self.coord_system == CoordinatesSystemType.CYLINDRICAL and i_dim == 1:
        #             geom_coeff = 1.0 / self.spatial_sym_var[0]
        #         MMS_source[i_var] = MMS_source[i_var] + geom_coeff * (sp.diff(F_c[i_var][i_dim],self.spatial_sym_var[i_dim]) - sp.diff(F_d[i_var][i_dim],self.spatial_sym_var[i_dim]))
        #     MMS_source[i_var] = MMS_source[i_var] - S[i_var][0]
        #     if self.isSteady == False:
        #         MMS_source[i_var] = MMS_source[i_var] + sp.diff(U[i_var],self.temporal_sym_var[0])
        # return MMS_source
        # MMS_conv_source = self.computeMMSConvectiveSourceTerm(num_params)
        # MMS_diff_source = self.computeMMSDiffusiveSourceTerm(num_params)
        # MMS_src_source = self.computeMMSSourceSourceTerm(num_params)
        # MMS_unsteady_source = self.computeMMSUnsteadySourceTerm(num_params)
        # src_tot = [MMS_conv_source[i]+MMS_diff_source[i]+MMS_src_source[i]+MMS_unsteady_source[i] for i in range(len(MMS_conv_source))]
        # return src_tot
    
        U = self.solVector(num_params)
        F_c = self.convectiveFlux(num_params)
        F_d = self.diffusiveFlux(num_params, True)
        S = self.sourceTerm(num_params, True)
        MMS_source = [0] * self.manuf_sol_container.getNumberFields()
        for i_var in range(self.manuf_sol_container.getNumberFields()):
            MMS_source[i_var] = 0
            geom_coeff = 1.0
            for i_dim in range(self.domain_dim):
                if self.coord_system == CoordinatesSystemType.CYLINDRICAL and i_dim == 1:
                    geom_coeff = 1.0 / self.spatial_sym_var[0]
                MMS_source[i_var] = MMS_source[i_var] + geom_coeff * (sp.diff(F_c[i_var][i_dim],self.spatial_sym_var[i_dim]) - sp.diff(F_d[i_var][i_dim],self.spatial_sym_var[i_dim]))
            MMS_source[i_var] = MMS_source[i_var] - S[i_var][0]
            if self.isSteady == False:
                MMS_source[i_var] = MMS_source[i_var] + sp.diff(U[i_var],self.temporal_sym_var[0])
        return MMS_source