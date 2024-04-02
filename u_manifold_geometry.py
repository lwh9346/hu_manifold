from sympy import Symbol, Eq, tanh, Or, And, Not, Gt, Lt, Abs

import numpy as np

from modulus.sym.geometry import Parameterization
from modulus.sym.geometry.primitives_3d import Box, Plane
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer
from modulus.sym.hydra import ModulusConfig
import time


# 定义几何
class MU(object):
    def __init__(
        self,
        nd: NonDimensionalizer,
        parameterized: bool = False,
    ):
        print(
            time.strftime(
                "[%H:%M:%S] - Start building geometry", time.localtime(time.time())
            )
        )
        # 以下参数都是在完整模型下的值，需要在对称模型中减半的值会有后缀_S
        # 几何尺寸
        # 均已无量纲化
        # 整个微流道
        total_length = nd.ndim(quantity(300, "um"))  # 长度
        total_width_S = nd.ndim(quantity(50, "um"))  # 宽度
        total_height = nd.ndim(quantity(400, "um"))  # 高度
        # 微流道部分
        microchannel_unit_width_S = nd.ndim(
            quantity(50, "um")
        )  # 一个微流道单元，W_c+W_w
        microchannel_width_S = Symbol("W_c")  # 流体通过部分宽度
        microchannel_wall_width_S = (
            microchannel_unit_width_S - microchannel_width_S
        )  # 微流道壁面厚度
        microchannel_total_height = nd.ndim(quantity(300, "um"))  # 微流道高度
        microchannel_fluid_height = Symbol("H_c")  # 微流道流体部分高度
        microchannel_thickness = (
            microchannel_total_height - microchannel_fluid_height
        )  # 微流道底部厚度
        # 歧管部分
        manifold_inlet_width_S = Symbol("W_mi")  # 歧管入口
        manifold_outlet_width_S = Symbol("W_mo")  # 歧管出口
        manifold_plate_height = nd.ndim(quantity(100, "um"))
        # 参数范围
        param_ranges = {
            microchannel_width_S: (
                nd.ndim(quantity(15, "um")),
                nd.ndim(quantity(35, "um")),
            ),
            manifold_inlet_width_S: (
                nd.ndim(quantity(100, "um")),
                nd.ndim(quantity(200, "um")),
            ),
            manifold_outlet_width_S: (
                nd.ndim(quantity(100, "um")),
                nd.ndim(quantity(200, "um")),
            ),
            microchannel_fluid_height: (
                nd.ndim(quantity(150, "um")),
                nd.ndim(quantity(200, "um")),
            ),
        }
        fixed_param_ranges = {
            microchannel_width_S: nd.ndim(quantity(30, "um")),
            manifold_inlet_width_S: nd.ndim(quantity(100, "um")),
            manifold_outlet_width_S: nd.ndim(quantity(200, "um")),
            microchannel_fluid_height: nd.ndim(quantity(200, "um")),
        }
        eps = 1e-9
        if parameterized:
            pr = Parameterization(param_ranges)
            self.pr = param_ranges
        else:
            pr = Parameterization(fixed_param_ranges)
            self.pr = fixed_param_ranges
        # 几何构建
        solid_bottom = Box(
            (0, 0, 0),
            (total_width_S / 2, total_length, microchannel_thickness),
            parameterization=pr,
        )
        solid_fin = Box(
            (0, 0, microchannel_thickness),
            (microchannel_wall_width_S / 2, total_length, microchannel_total_height),
            parameterization=pr,
        )
        solid_manifold = Box(
            (0, manifold_inlet_width_S / 2, microchannel_total_height),
            (
                total_width_S / 2,
                total_length - manifold_outlet_width_S / 2,
                total_height,
            ),
            parameterization=pr,
        )
        # 外部几何接口
        self.solid = solid_bottom + solid_fin + solid_manifold
        self.fluid = (
            Box(
                (0, 0, 0),
                (total_width_S / 2, total_length, total_height),
                parameterization=pr,
            )
            - self.solid
        )
        # 边界条件
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        self.on_boundary_symmetry_yz = Or(
            Lt(Abs(x - 0), eps), Lt(Abs(x - total_width_S / 2), eps)
        )
        self.on_boundary_symmetry_xz = Or(
            Lt(Abs(y - 0), eps), Lt(Abs(y - total_length), eps)
        )
        self.on_boundary_bottom = Lt(Abs(z - 0), eps)
        self.on_boundary_top = Lt(Abs(z - total_height), eps)
        self.on_boundary_inlet = And(
            Lt(y, manifold_inlet_width_S / 2), self.on_boundary_top
        )
        self.on_boundary_outlet = And(
            Gt(y, total_length - manifold_outlet_width_S / 2), self.on_boundary_top
        )
        # 用于计算的其他数值
        self.total_width_S = total_width_S
        print(time.strftime("[%H:%M:%S] - Geometry built", time.localtime(time.time())))


if __name__ == "__main__":
    from modulus.sym.utils.io.vtk import var_to_polyvtk

    geo = MU(nd=NonDimensionalizer(length_scale=quantity(1000, "um")))
    sb = geo.solid.sample_boundary(100000)
    var_to_polyvtk(sb, "visualize/solid_boundary")
    si = geo.solid.sample_interior(100000)
    var_to_polyvtk(si, "visualize/solid_interior")
    fb = geo.fluid.sample_boundary(100000)
    var_to_polyvtk(fb, "visualize/fluid_boundary")
    fi = geo.fluid.sample_interior(100000)
    var_to_polyvtk(fi, "visualize/fluid_interior")
    inlet = geo.fluid.sample_boundary(10000, criteria=geo.on_boundary_inlet)
    var_to_polyvtk(inlet, "visualize/inlet")
    outlet = geo.fluid.sample_boundary(10000, criteria=geo.on_boundary_outlet)
    var_to_polyvtk(outlet, "visualize/outlet")
