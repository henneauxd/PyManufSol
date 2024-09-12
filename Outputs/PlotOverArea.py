from BoundaryConditions.BoundaryGeometry import GeneralBoundaryGeometry, BoundaryGeometryFromPoints, BoundaryGeometryFromEquation
import sympy as sp
from Common.Utilities import *

class PlotOver2DArea:

    def __init__(self, sym_variables: list[sp.Expr], interfaces_eqs: Optional[list[GeneralBoundaryGeometry]] = None):
        if len(sym_variables) != 2:
            raise ValueError("The class PlotOver2DArea only valid for 2-D geometry.")
        self.sym_variables = sym_variables
        self.interfaces_eqs = interfaces_eqs
        if isinstance(self.interfaces_eqs, list):
            for i in self.interfaces_eqs:
                sym_var_interface = i.getSymVariables()
                if len(sym_var_interface) != len(self.sym_variables):
                    raise ValueError("The symbolic spatial variables of the interface eqs must have the same size as the ones used to define the 2-D area.")
                if any(a!=b for a,b in zip(sym_var_interface,self.sym_variables)):
                    raise ValueError("The symbolic spatial variables of the interface eqs must be the same as the ones used to define the 2-D area.")
                # if not is_linear(i.getBoundaryEquation(), self.sym_variables):
                #     raise ValueError("This class only able to deal with linear (straight line) interfaces.")

    def getAreaBoundaries(self) -> list[GeneralBoundaryGeometry]:
        raise NotImplementedError()

    def getMaxNbIntersectionsInterfaceBoundaries(self) -> int:
        raise NotImplementedError()

    def getInterfaces(self) -> list[GeneralBoundaryGeometry]:
        if isinstance(self.interfaces_eqs, list):
            return self.interfaces_eqs
        else:
            return []

    def getIntersectionsPtsBetweenBoundariesAndPlotLine(self, plot_line: sp.Expr) -> list[list[float]]:
        list_pts = []
        for bnd in self.getAreaBoundaries():
            pt = findIntersectionBetweenTwo2DLines(plot_line, bnd.getBoundaryEquation(), self.sym_variables)
            if len(pt) == 2:
                list_pts.append(pt)
        if len(list_pts) != self.getMaxNbIntersectionsInterfaceBoundaries():
            raise ValueError("The number of intersection points btw the line along which to plot and the 2-D area is %d and not %d."%(len(list_pts), self.getMaxNbIntersectionsInterfaceBoundaries()))
        return list_pts

    def getIntersectionsPtsBetweenInterfacesAndPlotLine(self, plot_line: sp.Expr) -> list[list[float]]:
        list_pts = []
        for bnd in self.getInterfaces():
            pt = findIntersectionBetweenTwo2DLines(plot_line, bnd.getBoundaryEquation(), self.sym_variables)
            if len(pt) == 2:
                list_pts.append(pt)
        return list_pts

    def plotThisFieldAlongThoseStraightLines(self, field: sp.Expr, plot_lines: list[sp.Expr], nb_pts: int, sideInterface: int, sampleDir: int, name_var: str, line_color: Optional[str] = None, line_type: Optional[str] = None):
        for line in plot_lines:
            pts_bnd = self.getIntersectionsPtsBetweenBoundariesAndPlotLine(line)
            pt_int = self.getIntersectionsPtsBetweenInterfacesAndPlotLine(line)
            pt_bnd = pts_bnd[0]
            interface = self.getInterfaces()
            if len(interface) > 0 and len(pt_int) > 0:
                pos_pt_wrt_int = interface[0].getBoundaryEquation().subs({self.sym_variables[0]: pt_bnd[0], self.sym_variables[1]: pt_bnd[1]})
                if (sideInterface < 0 and pos_pt_wrt_int > 0) or (sideInterface > 0 and pos_pt_wrt_int < 0):
                    pt_bnd = pts_bnd[1]
                plotVarBtwTwoPts(field, line, pt_bnd, pt_int[0], nb_pts, self.sym_variables, sampleDir, name_var, line_color, line_type)
            else:
                plotVarBtwTwoPts(field, line, pts_bnd[0], pts_bnd[1], nb_pts, self.sym_variables, sampleDir, name_var, line_color, line_type)

    def getAreaExtremities(self) -> list[list[float]]:
        raise NotImplementedError()

    def evalField(self, x: float, y: float, field: sp.Expr, sideInterface: int):
        field_eval = float("inf")
        interface = self.getInterfaces()
        if isinstance(field, list):
            field = field[0]
        if len(interface) > 0:
            pos_pt_wrt_int = interface[0].getBoundaryEquation().subs({self.sym_variables[0]: x, self.sym_variables[1]: y})
            if (sideInterface < 0 and pos_pt_wrt_int <= 0) or (sideInterface > 0 and pos_pt_wrt_int >= 0):
                field_eval = field.subs({self.sym_variables[0]: x, self.sym_variables[1]: y})
        else:
            field_eval = field.subs({self.sym_variables[0]: x, self.sym_variables[1]: y})
        return field_eval

    def plotThisFieldOver2DArea(self, field: sp.Expr, nb_pts: int, sideInterface: int, ax, name_var: str):
        area_extr = self.getAreaExtremities()
        x = np.linspace(area_extr[0][0], area_extr[1][0], nb_pts)
        y = np.linspace(area_extr[0][1], area_extr[1][1], nb_pts)
        X, Y = np.meshgrid(x, y)
        z = np.zeros((np.size(x), np.size(y)))
        interface = self.getInterfaces()
        int_pos = np.zeros((np.size(x), np.size(y)))         
        for i in range(len(x)):
            for j in range(len(y)):
                z[i,j] = self.evalField(x[i], y[j], field, sideInterface)
                # int_pos[i,j] = interface[0].getBoundaryEquation().subs({self.sym_variables[0]: x[i], self.sym_variables[1]: y[j]})
        # z = np.array(self.evalField(np.ravel(x), np.ravel(y), field, sideInterface))
        Z = np.transpose(z.reshape(X.shape))
        int_pos = np.transpose(int_pos.reshape(X.shape))
        Z_trunc = np.ma.masked_where(Z==float("inf"), Z)
        mySurf = ax.plot_surface(X, Y, Z_trunc, label="%s"%name_var)
        mySurf._facecolors2d=mySurf._facecolor3d
        mySurf._edgecolors2d=mySurf._edgecolor3d
        # plt.contour(X,Y,int_pos,levels=[0],linewidth=3)
        # plt.contourf(X,Y,Z,20)

    def plotThisVectorOver2DArea(self, field: sp.Expr, nb_pts: int, sideInterface: int, ax, name_var: str):
        area_extr = self.getAreaExtremities()
        x = np.linspace(area_extr[0][0], area_extr[1][0], nb_pts)
        y = np.linspace(area_extr[0][1], area_extr[1][1], nb_pts)
        X, Y = np.meshgrid(x, y)
        z1 = np.zeros((np.size(x), np.size(y)))
        z2 = np.zeros((np.size(x), np.size(y)))
        interface = self.getInterfaces()
        int_pos = np.zeros((np.size(x), np.size(y)))        
        for i in range(len(x)):
            for j in range(len(y)):
                z1[i,j] = self.evalField(x[i], y[j], field[0], sideInterface)
                z2[i,j] = self.evalField(x[i], y[j], field[1], sideInterface)
                int_pos[i,j] = interface[0].getBoundaryEquation().subs({self.sym_variables[0]: x[i], self.sym_variables[1]: y[j]})
        # z = np.array(self.evalField(np.ravel(x), np.ravel(y), field, sideInterface))
        Z1 = np.transpose(z1.reshape(X.shape))
        Z2 = np.transpose(z2.reshape(X.shape))
        int_pos = np.transpose(int_pos.reshape(X.shape))
        Z1_trunc = np.ma.masked_where(Z1==float("inf"), Z1)
        Z2_trunc = np.ma.masked_where(Z2==float("inf"), Z2)
        norm_vec = np.sqrt(Z1_trunc**2 + Z2_trunc**2)
        Z1_trunc_norm = Z1_trunc / norm_vec
        Z2_trunc_norm = Z2_trunc / norm_vec
        # ax.quiver(X, Y, Z1, Z2, angles='xy', scale_units='xy', units='width', scale=1)
        myquiver = ax.quiver(X, Y, Z1_trunc_norm, Z2_trunc_norm, norm_vec, units='width',scale=50)
        myquiver.set_cmap('coolwarm')
        # cbar = plt.colorbar(myquiver)

        plt.contour(X,Y,int_pos,levels=[0])

class PlotOver2DAreaWithStraightBoundaries(PlotOver2DArea):

    def __init__(self, sym_variables: list[sp.Expr], interfaces_eqs: list[GeneralBoundaryGeometry], pts_bnd: list[list[float]]):
        super().__init__(sym_variables, interfaces_eqs)
        if len(pts_bnd) < 3:
            raise ValueError("To define the area, at least 3 control points are required but only %d were provided."%(len(pts_bnd)))
        self.pts_bnd = pts_bnd

    def getAreaBoundaries(self) -> list[GeneralBoundaryGeometry]:
        list_bnd = []
        for i in range(len(self.pts_bnd)):
            if i == len(self.pts_bnd)-1:
                two_pts = [self.pts_bnd[i], self.pts_bnd[0]]
            else:
                two_pts = [self.pts_bnd[i], self.pts_bnd[i+1]]
            line_geo = BoundaryGeometryFromPoints(self.sym_variables, two_pts, "")
            list_bnd.append(line_geo)
        return list_bnd

    def getMaxNbIntersectionsInterfaceBoundaries(self) -> int:
        return 2

    def getAreaExtremities(self) -> list[list[float]]:
        inf = float("inf")
        x_min = inf
        x_max = -inf
        y_min = inf
        y_max = -inf
        for i in range(len(self.pts_bnd)):
            x_min = min(x_min, self.pts_bnd[i][0])
            x_max = max(x_max, self.pts_bnd[i][0])
            y_min = min(x_min, self.pts_bnd[i][1])
            y_max = max(x_max, self.pts_bnd[i][1])
        return [[x_min, y_min], [x_max, y_max]]

class PlotOver2DCircularAreaCenteredAtOrigin(PlotOver2DArea):

    def __init__(self, sym_variables: list[sp.Expr], interfaces_eqs: list[GeneralBoundaryGeometry], radius_circle: float):
        super().__init__(sym_variables, interfaces_eqs)
        self.radius_circle = radius_circle

    def getAreaBoundaries(self) -> list[GeneralBoundaryGeometry]:
        list_bnd = []
        list_bnd.append(BoundaryGeometryFromEquation(self.sym_variables, [1.0, 0.0, self.radius_circle], ""))
        return list_bnd

    def getMaxNbIntersectionsInterfaceBoundaries(self) -> int:
        return 1

    def getIntersectionsPtsBetweenBoundariesAndPlotLine(self, plot_line: sp.Expr) -> list[list[float]]:
        list_pts = super().getIntersectionsPtsBetweenBoundariesAndPlotLine(plot_line)
        list_pts.append([0.0, 0.0])
        return list_pts

    def getAreaExtremities(self) -> list[list[float]]:
        r_min = 0.0
        r_max = self.radius_circle
        theta_min = 0.0
        theta_max = 360.0
        return [[r_min, theta_min], [r_max, theta_max]]
