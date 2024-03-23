from sympy import Symbol, Eq, tanh, Or, And, Not, Gt, Lt, Abs

import numpy as np

from modulus.sym.geometry import Parameterization
from modulus.sym.geometry.primitives_3d import Box, Plane
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer
import time

# 几何参数
# 以下参数都是在1/4模型下的值，和文章中有出入

# 几何尺寸
# 均已无量纲化
_nd = NonDimensionalizer(length_scale=quantity(1000, "um"))
# 整个微流道
channel_length = _nd.ndim(quantity(1500, "um"))  # 长度
SMM_width = _nd.ndim(quantity(300, "um"))  # 宽度
total_height = _nd.ndim(quantity(800, "um"))  # 高度
# 微流道部分
microchannel_unit_width = _nd.ndim(quantity(50, "um"))  # 一个微流道单元，W_c+W_w
microchannel_width = Symbol("W_c")  # 流体通过部分宽度，文中是没有参数化的，这边给加上
microchannel_wall_width = microchannel_unit_width - microchannel_width  # 微流道壁面厚度
microchannel_height = _nd.ndim(quantity(200, "um"))  # 微流道高度
microchannel_bottom_thickness = _nd.ndim(quantity(100, "um"))  # 微流道底部厚度
# 歧管部分
manifold_inlet_width = Symbol("W_mi")  # 歧管入口
manifold_inlet_height = _nd.ndim(quantity(300, "um"))
manifold_wall_width = _nd.ndim(quantity(100, "um"))  # 出入口之间的隔板
manifold_outlet_width = SMM_width - manifold_inlet_width - manifold_wall_width  # 出口
manifold_plate_width = Symbol("W_mp")  # 射流板的宽度
manifold_plate_height = _nd.ndim(quantity(200, "um"))
manifold_height = _nd.ndim(quantity(500, "um"))  # 歧管总高
# 参数范围
param_ranges = {
    microchannel_width: (_nd.ndim(quantity(10, "um")), _nd.ndim(quantity(30, "um"))),
    manifold_inlet_width: (_nd.ndim(quantity(50, "um")), _nd.ndim(quantity(150, "um"))),
    manifold_plate_width: _nd.ndim(quantity(0, "um")),
}
fixed_param_ranges = {
    microchannel_width: _nd.ndim(quantity(25, "um")),
    manifold_inlet_width: _nd.ndim(quantity(50, "um")),
    manifold_plate_width: _nd.ndim(quantity(100, "um")),
}
eps = 1e-9


# 定义几何
class SMM(object):
    def __init__(self, parameterized: bool = False):
        # 参数范围
        print(
            time.strftime(
                "[%H:%M:%S] - Start building geometry", time.localtime(time.time())
            )
        )
        if parameterized:
            pr = Parameterization(param_ranges)
            self.pr = param_ranges
        else:
            pr = Parameterization(fixed_param_ranges)
            self.pr = fixed_param_ranges
        # 微流道部分
        microchannel_fluid = [
            Box(
                (
                    microchannel_wall_width / 2 + index * microchannel_unit_width,
                    0,
                    microchannel_bottom_thickness,
                ),
                (
                    microchannel_wall_width / 2
                    + microchannel_width
                    + index * microchannel_unit_width,
                    SMM_width,
                    microchannel_bottom_thickness + microchannel_height,
                ),
                parameterization=pr,
            )
            for index in range(30)
        ]
        microchannel_fluid = sum(microchannel_fluid[1:], start=microchannel_fluid[0])
        microchannel_solid = (
            Box(
                (0, 0, 0),
                (
                    channel_length,
                    SMM_width,
                    microchannel_height + microchannel_bottom_thickness,
                ),
                parameterization=pr,
            )
            - microchannel_fluid
        )
        # 歧管部分
        manifold_origin = (0, 0, microchannel_bottom_thickness + microchannel_height)
        manifold_solid = (
            Box(  # 出入口隔板
                (
                    manifold_origin[0],
                    manifold_origin[1]
                    + SMM_width
                    - manifold_inlet_width
                    - manifold_wall_width,
                    manifold_origin[2],
                ),
                (
                    manifold_origin[0] + channel_length,
                    manifold_origin[1] + SMM_width - manifold_inlet_width,
                    manifold_origin[2] + manifold_height,
                ),
                parameterization=pr,
            )
            + Box(  # 入口顶盖
                (
                    manifold_origin[0],
                    manifold_origin[1] + SMM_width - manifold_inlet_width,
                    manifold_origin[2] + manifold_inlet_height,
                ),
                (
                    manifold_origin[0] + channel_length,
                    manifold_origin[1] + SMM_width,
                    manifold_origin[2] + manifold_height,
                ),
                parameterization=pr,
            )
            + Box(  # 射流板
                (
                    manifold_origin[0],
                    manifold_origin[1]
                    + SMM_width
                    - manifold_inlet_width
                    - manifold_wall_width
                    - manifold_plate_width,
                    manifold_origin[2],
                ),
                (
                    manifold_origin[0] + channel_length,
                    manifold_origin[1] + manifold_outlet_width,
                    manifold_origin[2] + manifold_plate_height,
                ),
                parameterization=pr,
            )
        )
        manifold_fluid = (
            Box(
                manifold_origin,
                (
                    manifold_origin[0] + channel_length,
                    manifold_origin[1] + SMM_width,
                    manifold_origin[2] + manifold_height,
                ),
                parameterization=pr,
            )
            - manifold_solid
        )
        # 流体域与固体域
        self.solid = manifold_solid + microchannel_solid
        self.fluid = manifold_fluid + microchannel_fluid
        # 出入口
        self.inlet = Plane(
            (
                0,
                SMM_width - manifold_inlet_width,
                microchannel_height + microchannel_bottom_thickness,
            ),
            (
                0,
                SMM_width,
                microchannel_height
                + microchannel_bottom_thickness
                + manifold_inlet_height,
            ),
            parameterization=pr,
        )
        outlet_rotate_point = (0, 0, total_height)
        self.outlet = Plane(
            (
                outlet_rotate_point[0],
                outlet_rotate_point[1],
                outlet_rotate_point[2] - channel_length,
            ),
            (
                outlet_rotate_point[0],
                outlet_rotate_point[1] + manifold_outlet_width,
                outlet_rotate_point[2],
            ),
            parameterization=pr,
        ).rotate(np.pi / 2, "y", outlet_rotate_point)
        # 额外的积分连续性平面
        microchannel_intergal_plane = [
            Plane(
                (
                    SMM_width / 2,
                    microchannel_wall_width / 2 + index * microchannel_unit_width,
                    microchannel_bottom_thickness,
                ),
                (
                    SMM_width / 2,
                    microchannel_wall_width / 2
                    + microchannel_width
                    + index * microchannel_unit_width,
                    microchannel_bottom_thickness + microchannel_height,
                ),
                parameterization=pr,
            )
            for index in range(30)
        ]
        microchannel_intergal_plane = sum(
            microchannel_intergal_plane[1:], start=microchannel_intergal_plane[0]
        )
        self.microchannel_intergal_plane = microchannel_intergal_plane.rotate(
            -np.pi / 2, "z", (SMM_width / 2, SMM_width / 2, 0)
        )
        # 边界条件
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        self.on_boundary_inlet = And(
            Lt(Abs(x - 0), eps),
            Gt(y, SMM_width - manifold_inlet_width),
            Gt(z, microchannel_bottom_thickness + microchannel_height),
            # Lt(y, SMM_width), 不可能
            Lt(
                z,
                microchannel_bottom_thickness
                + microchannel_height
                + manifold_inlet_height,
            ),
        )
        self.on_boundary_outlet = And(
            Lt(Abs(z - total_height), eps),
            Lt(y, manifold_outlet_width),
        )
        self.on_symmetry_xz = Or(Lt(Abs(y - 0), eps), Lt(Abs(y - SMM_width), eps))
        self.on_symmetry_yz = Lt(Abs(x - channel_length), eps)
        print(time.strftime("[%H:%M:%S] - Geometry built", time.localtime(time.time())))


if __name__ == "__main__":
    from modulus.sym.utils.io.vtk import var_to_polyvtk

    smm = SMM()
    sb = smm.solid.sample_boundary(100000)
    var_to_polyvtk(sb, "visualize/solid_boundary")
    si = smm.solid.sample_interior(100000)
    var_to_polyvtk(si, "visualize/solid_interior")
    fb = smm.fluid.sample_boundary(100000)
    var_to_polyvtk(fb, "visualize/fluid_boundary")
    fi = smm.fluid.sample_interior(100000)
    var_to_polyvtk(fi, "visualize/fluid_interior")
    inlet = smm.inlet.sample_boundary(10000)
    var_to_polyvtk(inlet, "visualize/inlet")
    outlet = smm.outlet.sample_boundary(10000)
    var_to_polyvtk(outlet, "visualize/outlet")
    mcip = smm.microchannel_intergal_plane.sample_boundary(10000)
    var_to_polyvtk(mcip, "visualize/mcip")
