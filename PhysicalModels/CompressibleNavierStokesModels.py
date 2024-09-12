from PhysicalModels.ThermoPhysicalModels.EOS_factory import MieGruneisenEOS
from PhysicalModels.ThermoPhysicalModels.TransportProperties_factory import FluidTransportProperty
from PhysicalModels.GeneralPhysicalModels import GeneralPhysicalModels
from ManufSolution.GeneralManufSol import assembleManufSolListIntoTagDic 
from ManufSolution.ManufSolContainer.GeneralManufSolContainer import GeneralManufSolContainerSinglePhase
from ManufSolution.ManufSolContainer.CompressibleFlowManufSolContainer import CompressibleFlowManufSolContainer
from Common.MMSTags import CompressibleFlowVarSetTags, SolutionTags, SolutionGradientTags
from Common.MMSTags import FluidSolutionTags as fst
from Common.MMSTags import FluidSolutionGradientTags as fsgt
from Common.MMSTags import CoordinatesSystemType
from Common.Utilities import *
from typing import Optional

class CompressibleNavierStokesModels(GeneralPhysicalModels):
    def __init__(self, tag: str, manuf_sol_container: GeneralManufSolContainerSinglePhase, domain_dim: int, eos: MieGruneisenEOS, transport_prop: FluidTransportProperty, coord_system: CoordinatesSystemType = CoordinatesSystemType.CARTESIAN):
        super().__init__(tag, manuf_sol_container, domain_dim, coord_system)
        if not isinstance(manuf_sol_container, CompressibleFlowManufSolContainer):
            raise ValueError("The manuf container passed as args should be of 'CompressibleFlowManufSolContainer' type.")
        self.eos = eos
        self.transport_prop = transport_prop
        
    def solVector(self, num_params: Optional[dict] = None):
        list_var = [0] * self.manuf_sol_container.getNumberFields()
        if self.manuf_sol_container.getVarSetChoice() == CompressibleFlowVarSetTags.CONSERVATIVE:
            list_var = list(self.manuf_sol_container.getDicManufSolPerTag().values())
        elif self.manuf_sol_container.getVarSetChoice() == CompressibleFlowVarSetTags.PRIMITIVE_PVT:
            list_var = self.eos.primitivePVTToConservative(self.manuf_sol_container.getDicManufSolPerTag())
        elif self.manuf_sol_container.getVarSetChoice() == CompressibleFlowVarSetTags.PRIMITIVE_PVRHO:
            list_var = self.eos.primitivePVRHOToConservative(self.manuf_sol_container.getDicManufSolPerTag())
        return subsNumParams(list_var, num_params)

    def convertFromThisVarSetToThisOther(self, manuf_sol_dic: dict, current_var_set: CompressibleFlowVarSetTags, new_var_set: CompressibleFlowVarSetTags):
        if current_var_set == new_var_set:
            return manuf_sol_dic
        tag_new_var_set = self.manuf_sol_container.getSolsTagsFromVarSetChoice(new_var_set)
        if current_var_set == CompressibleFlowVarSetTags.CONSERVATIVE:
            if new_var_set == CompressibleFlowVarSetTags.PRIMITIVE_PVT:
                sol_list_prim = self.eos.conservativeToPrimitivePVT(manuf_sol_dic)
                return assembleManufSolListIntoTagDic(sol_list_prim, tag_new_var_set)
            elif new_var_set == CompressibleFlowVarSetTags.PRIMITIVE_PVRHO:
                sol_list_prim = self.eos.conservativeToPrimitivePVRHO(manuf_sol_dic)
                return assembleManufSolListIntoTagDic(sol_list_prim, tag_new_var_set)
        elif current_var_set == CompressibleFlowVarSetTags.PRIMITIVE_PVT:
            if new_var_set == CompressibleFlowVarSetTags.CONSERVATIVE:
                sol_list_cons = self.eos.primitivePVTToConservative(manuf_sol_dic)
                return assembleManufSolListIntoTagDic(sol_list_cons, tag_new_var_set)
            elif new_var_set == CompressibleFlowVarSetTags.PRIMITIVE_PVRHO:
                sol_list_cons = self.eos.primitivePVTToprimitivePVRHO(manuf_sol_dic)
                return assembleManufSolListIntoTagDic(sol_list_cons, tag_new_var_set)
        elif current_var_set == CompressibleFlowVarSetTags.PRIMITIVE_PVRHO:
            if new_var_set == CompressibleFlowVarSetTags.CONSERVATIVE:
                sol_list_cons = self.eos.primitivePVRHOToConservative(manuf_sol_dic)
                return assembleManufSolListIntoTagDic(sol_list_cons, tag_new_var_set)
            elif new_var_set == CompressibleFlowVarSetTags.PRIMITIVE_PVT:
                sol_list_cons = self.eos.primitivePVRHOToprimitivePVT(manuf_sol_dic)
                return assembleManufSolListIntoTagDic(sol_list_cons, tag_new_var_set)

    def getSolTag(self, tag: SolutionTags, num_params: Optional[dict] = None):
        if not isinstance(tag, fst):
            raise ValueError("The solution tag should be of 'FluidSolutionTags' type")
        var = 0.0

        manuf_sol_dic = self.convertFromThisVarSetToThisOther(self.manuf_sol_container.getDicManufSolPerTag(), self.manuf_sol_container.getVarSetChoice(), CompressibleFlowVarSetTags.PRIMITIVE_PVT)
        tag_prim_pvt = self.manuf_sol_container.getSolsTagsFromVarSetChoice(CompressibleFlowVarSetTags.PRIMITIVE_PVT)
        # if self.manuf_sol_container.getVarSetChoice() == CompressibleFlowVarSetTags.CONSERVATIVE:
        #     # sol_list_cons = list(self.manuf_sol_container.getDicManufSolPerTag().values())
        #     sol_list_prim = self.eos.conservativeToPrimitivePVT(manuf_sol_dic)
        #     manuf_sol_dic = assembleManufSolListIntoTagDic(sol_list_prim, tag_prim_pvt)

        if tag in tag_prim_pvt:
            var = manuf_sol_dic[tag]
        else:
            p = self.getSolTag(fst.PRESSURE, num_params)
            T = self.getSolTag(fst.TEMPERATURE, num_params)
            vel = [self.getSolTag(fst.VELOCITY_X, num_params), self.getSolTag(fst.VELOCITY_Y, num_params)]
            if self.domain_dim == 3:
                vel.append(self.getSolTag(fst.VELOCITY_Z, num_params))
            if tag == fst.DENSITY:
                var = self.eos.rho_pT(p,T)
            elif tag == fst.VELOCITY_VEC:
                var = vel
            elif tag == fst.MOMENTUM_X:
                var = self.getSolTag(fst.DENSITY, num_params) * vel[0]
            elif tag == fst.MOMENTUM_Y:
                var = self.getSolTag(fst.DENSITY, num_params) * vel[1]
            elif tag == fst.MOMENTUM_Z:
                var = self.getSolTag(fst.DENSITY, num_params) * vel[2]
            elif tag == fst.MOMENTUM_VEC:
                mom_vec = [self.getSolTag(fst.MOMENTUM_X, num_params), self.getSolTag(fst.MOMENTUM_Y, num_params)]
                if self.domain_dim == 3:
                    mom_vec.append(self.getSolTag(fst.MOMENTUM_Z, num_params))
                var = mom_vec
            elif tag == fst.INTERNALENERGY:
                var = self.eos.e_rhoT(self.getSolTag(fst.DENSITY, num_params), T)
            elif tag == fst.KINETICENERGY:
                for i in vel:
                    var = var + 0.5 * i*i
            elif tag == fst.TOTALENERGY:
                var = self.getSolTag(fst.INTERNALENERGY, num_params) + self.getSolTag(fst.KINETICENERGY, num_params)
            elif tag == fst.RHOTOTALENERGY:
                var = self.getSolTag(fst.DENSITY, num_params) * self.getSolTag(fst.TOTALENERGY)
            elif tag == fst.ENTHALPY:
                var = self.getSolTag(fst.INTERNALENERGY, num_params) + self.getSolTag(fst.PRESSURE, num_params) / self.getSolTag(fst.DENSITY, num_params)
            elif tag == fst.TOTALENTHALPY:
                var = self.getSolTag(fst.TOTALENERGY, num_params) + self.getSolTag(fst.PRESSURE, num_params) / self.getSolTag(fst.DENSITY, num_params)
            elif tag == fst.CONVECTIVEFLUX_X1:
                F_c = self.convectiveFlux(num_params)
                var = F_c[0][0]
            elif tag == fst.CONVECTIVEFLUX_X2:
                F_c = self.convectiveFlux(num_params)
                var = F_c[1][0]
            elif tag == fst.CONVECTIVEFLUX_X3:
                F_c = self.convectiveFlux(num_params)
                var = F_c[2][0]
            elif tag == fst.CONVECTIVEFLUX_X4:
                F_c = self.convectiveFlux(num_params)
                var = F_c[3][0]
            elif tag == fst.CONVECTIVEFLUX_Y1:
                F_c = self.convectiveFlux(num_params)
                var = F_c[0][1]
            elif tag == fst.CONVECTIVEFLUX_Y2:
                F_c = self.convectiveFlux(num_params)
                var = F_c[1][1]
            elif tag == fst.CONVECTIVEFLUX_Y3:
                F_c = self.convectiveFlux(num_params)
                var = F_c[2][1]
            elif tag == fst.CONVECTIVEFLUX_Y4:
                F_c = self.convectiveFlux(num_params)
                var = F_c[3][1]
            else:
                raise ValueError("The solution tag is unknown.")
        return subsNumParams(var, num_params)

    def getGradVarTag(self, tag: SolutionGradientTags, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        if not isinstance(tag, fsgt):
            raise ValueError("The gradient solution tag should be of 'FluidSolutionGradientTags' type")

        list_ind_var = self.spatial_sym_var#[[self.spatial_sym_var[i]] for i in range(self.domain_dim)]

        if tag == fsgt.GRADVEL:
            gradVel = self.getGradSolTag(fst.VELOCITY_VEC, list_ind_var, num_params, replace_params_before_deriv)
            return gradVel
        if tag == fsgt.VORTICITY:
            dVel = self.getGradVarTag(fsgt.GRADVEL, num_params, replace_params_before_deriv)
            return [dVel[1][0] - dVel[0][1]]
        elif tag == fsgt.DIVERGENCEVEL:
            divVel = self.getDivergenceSolTag(fst.VELOCITY_VEC, list_ind_var, num_params, replace_params_before_deriv)
            return divVel
        elif tag == fsgt.GRADT:
            dT = self.getGradSolTag(fst.TEMPERATURE, list_ind_var, num_params, replace_params_before_deriv)
            return dT
        elif tag == fsgt.SHEARSTRESS:
            dVel = self.getGradVarTag(fsgt.GRADVEL, num_params, replace_params_before_deriv)
            divVel = self.getGradVarTag(fsgt.DIVERGENCEVEL, num_params, replace_params_before_deriv)
            mu = self.transport_prop.getDynamicViscosity(self.solVector(num_params))
            tau = initListOfLists(self.domain_dim, self.domain_dim)
            for i_dim in range(self.domain_dim):
                for j_dim in range(self.domain_dim):
                    tau[i_dim][j_dim] = mu * (dVel[i_dim][j_dim] + dVel[j_dim][i_dim])
                    if i_dim == j_dim:
                        tau[i_dim][j_dim] = tau[i_dim][j_dim] - 2./3. * mu * divVel[0][0]
            return tau
        elif tag == fsgt.SHEARSTRESS_XX:
            tau = self.getGradVarTag(fsgt.SHEARSTRESS, num_params, replace_params_before_deriv)
            return tau[0][0]
        elif tag == fsgt.SHEARSTRESS_XY:
            tau = self.getGradVarTag(fsgt.SHEARSTRESS, num_params, replace_params_before_deriv)
            return tau[0][1]
        elif tag == fsgt.SHEARSTRESS_YY:
            tau = self.getGradVarTag(fsgt.SHEARSTRESS, num_params, replace_params_before_deriv)
            return tau[1][1]
        elif tag == fsgt.HEATFLUX:
            dT = self.getGradVarTag(fsgt.GRADT, num_params, replace_params_before_deriv)
            kappa = self.transport_prop.getThermalConductivity(self.solVector(num_params))
            q = [0.0] * self.domain_dim
            for i_dim in range(self.domain_dim):
                q[i_dim] = -kappa * dT[0][i_dim]
            return q
        elif tag == fsgt.HEATFLUX_X:
            q = self.getGradVarTag(fsgt.HEATFLUX, num_params, replace_params_before_deriv)
            return q[0]
        elif tag == fsgt.HEATFLUX_Y:
            q = self.getGradVarTag(fsgt.HEATFLUX, num_params, replace_params_before_deriv)
            return q[1]
        elif tag == fsgt.VISCOUSDISSIPATION:
            tau = self.getGradVarTag(fsgt.SHEARSTRESS, num_params, replace_params_before_deriv)
            vel = self.getSolTag(fst.VELOCITY_VEC, num_params)    
            tau_vel = multiplyNestedListsWithList(tau,vel)
            return tau_vel
        elif tag == fsgt.HEATFLUX_MINUS_VISCOUSDISSIPATION:
            q = self.getGradVarTag(fsgt.HEATFLUX, num_params, replace_params_before_deriv)
            tau_vel = self.getGradVarTag(fsgt.VISCOUSDISSIPATION, num_params, replace_params_before_deriv)
            return [a-b for a,b in zip(q,tau_vel)]
        elif tag == fsgt.DIFFUSIVEFLUX_X1:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[0][0]
        elif tag == fsgt.DIFFUSIVEFLUX_X2:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[1][0]
        elif tag == fsgt.DIFFUSIVEFLUX_X3:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[2][0]
        elif tag == fsgt.DIFFUSIVEFLUX_X4:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[3][0]
        elif tag == fsgt.DIFFUSIVEFLUX_Y1:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[0][1]
        elif tag == fsgt.DIFFUSIVEFLUX_Y2:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[1][1]
        elif tag == fsgt.DIFFUSIVEFLUX_Y3:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[2][1]
        elif tag == fsgt.DIFFUSIVEFLUX_Y4:
            F_d = self.diffusiveFlux(num_params, replace_params_before_deriv)
            return F_d[3][1]
        else:
            raise ValueError("The gradient variable tag is unknown.")

    def convectiveFlux(self, num_params: Optional[dict] = None):
        F_c = initListOfLists(self.manuf_sol_container.getNumberFields(), self.domain_dim)

        rho = self.getSolTag(fst.DENSITY, None)
        vel = self.getSolTag(fst.VELOCITY_VEC, None)
        P = self.getSolTag(fst.PRESSURE, None)
        E = self.getSolTag(fst.TOTALENERGY, None)

        # Continuity
        for i_dim in range(self.domain_dim):
            F_c[0][i_dim] = rho * vel[i_dim]    
        # Momentum
        for i_dim in range(self.domain_dim):
            for j_dim in range(self.domain_dim):
                F_c[i_dim+1][j_dim] = rho * vel[i_dim] * vel[j_dim] 
                if i_dim == j_dim:
                    F_c[i_dim+1][j_dim] = F_c[i_dim+1][j_dim] + P
        # Energy
        for i_dim in range(self.domain_dim):
            F_c[self.manuf_sol_container.getNumberFields()-1][i_dim] = rho * vel[i_dim] * E + vel[i_dim] * P

        return subsNumParams(F_c, num_params)

    def diffusiveFlux(self, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        F_d = initListOfLists(self.manuf_sol_container.getNumberFields(), self.domain_dim)

        vel = self.getSolTag(fst.VELOCITY_VEC, None)
        tau = self.getGradVarTag(fsgt.SHEARSTRESS, num_params, replace_params_before_deriv)
        q = self.getGradVarTag(fsgt.HEATFLUX, num_params, replace_params_before_deriv)

        # Continuity    
        for i_dim in range(self.domain_dim):
            F_d[0][i_dim] = 0.0 
        # Momentum
        for i_dim in range(self.domain_dim):
            for j_dim in range(self.domain_dim):
                F_d[i_dim+1][j_dim] = tau[i_dim][j_dim] 
        # Energy
        for i_dim in range(self.domain_dim):
            F_d[self.manuf_sol_container.getNumberFields()-1][i_dim] = -q[i_dim]
            for j_dim in range(self.domain_dim):
                F_d[self.manuf_sol_container.getNumberFields()-1][i_dim] = F_d[self.manuf_sol_container.getNumberFields()-1][i_dim] + vel[j_dim] * tau[i_dim][j_dim]

        return subsNumParams(F_d, num_params)

    def sourceTerm(self, num_params: Optional[dict] = None, replace_params_before_deriv: bool = False):
        S = initListOfLists(self.manuf_sol_container.getNumberFields(), 1, 0.0)
        if self.coord_system == CoordinatesSystemType.CYLINDRICAL:
            r = -self.spatial_sym_var[0]
            rho = self.getSolTag(fst.DENSITY, None)
            vel = self.getSolTag(fst.VELOCITY_VEC, None)
            H = self.getSolTag(fst.TOTALENTHALPY, None)
            tau = self.getGradVarTag(fsgt.SHEARSTRESS, num_params, replace_params_before_deriv)
            q = self.getGradVarTag(fsgt.HEATFLUX, num_params, replace_params_before_deriv)
            # Continuity 
            S[0][0] = 1./r * (rho*vel[0])
            # Momentum
            S[1][0] = 1./r * (rho*(vel[0]*vel[0]-vel[1]*vel[1]) - tau[0][0] + tau[1][1])
            S[2][0] = 1./r * (2.0*rho*vel[0]*vel[1] - 2.0*tau[1][0])
            if self.domain_dim == 3:
                S[3][0] = 1./r * (rho*vel[1]*vel[3] - tau[0][2])
            # Energy
            S[self.manuf_sol_container.getNumberFields()-1][0] = 1./r * (rho*vel[0]*H + q[0] - vel[0] * tau[0][0] - vel[1]*tau[1][0])
            if self.domain_dim == 3:
                S[self.manuf_sol_container.getNumberFields()-1][0] = S[self.manuf_sol_container.getNumberFields()-1][0] - 1./r * (vel[2] * tau[0][2])
        return S
        
