import os
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from sympy import Symbol, Eq, Abs, tanh, Or, And
import numpy as np

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box, Channel, Plane
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym import ureg

from hu_manifold_geometry import *

import wandb


@modulus.sym.main(config_path="conf", config_name="conf_flow_fix")
def run(cfg: ModulusConfig) -> None:
    # 初始化日志
    wandb_config = {
        "nondimensionalization": cfg.custom.nd,
        "batch_size": cfg.batch_size,
        "navier_stocks": cfg.custom.ns,
        "optimizer": cfg.optimizer,
    }
    wandb.init(project="hu_manifold", config=wandb_config)
    # 定义方程与特征量
    length_scale = quantity(cfg.custom.nd.length.value, cfg.custom.nd.length.unit)
    velocity_scale = quantity(cfg.custom.nd.velocity.value, cfg.custom.nd.velocity.unit)
    pressure_scale = quantity(cfg.custom.nd.pressure.value, cfg.custom.nd.pressure.unit)
    nu = quantity(cfg.custom.ns.nu.value, cfg.custom.ns.nu.unit)
    rho = quantity(cfg.custom.ns.rho.value, cfg.custom.ns.rho.unit)
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=pressure_scale / velocity_scale**2 * length_scale**3,
    )
    ns = NavierStokes(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    ns_nodes = ns.make_nodes()
    normal_dot_vel = NormalDotVec()
    # 网络架构
    input_keys = [Key("x"), Key("y"), Key("z")]
    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]
    flow_net = FourierNetArch(
        input_keys,
        output_keys,
        frequencies_params=[i for i in range(31)],
        skip_connections=True,
    )
    flow_nodes = (
        ns_nodes
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )
    geo = SMM()
    # 边界条件
    flow_domain = Domain()
    inlet_vol_flow = quantity(
        cfg.custom.bc.inlet_vol_flow.value, cfg.custom.bc.inlet_vol_flow.unit
    )
    inlet_continuity = IntegralBoundaryConstraint(  # 入口流量
        nodes=flow_nodes,
        geometry=geo.inlet,
        outvar={"normal_dot_vel": nd.ndim(inlet_vol_flow)},
        batch_size=5,
        integral_batch_size=cfg.batch_size.Inlet,
        parameterization=geo.pr,
        fixed_dataset=True,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(inlet_continuity, "inlet_continuity")
    outlet_continuity = IntegralBoundaryConstraint(  # 出口流量
        nodes=flow_nodes,
        geometry=geo.outlet,
        outvar={"normal_dot_vel": nd.ndim(inlet_vol_flow)},
        batch_size=5,
        integral_batch_size=cfg.batch_size.Outlet,
        parameterization=geo.pr,
        fixed_dataset=True,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(outlet_continuity, "outlet_continuity")
    internal_continuity = IntegralBoundaryConstraint(  # 横向流量
        nodes=flow_nodes,
        geometry=geo.microchannel_intergal_plane,
        outvar={"normal_dot_vel": nd.ndim(inlet_vol_flow)},
        batch_size=5,
        integral_batch_size=cfg.batch_size.MCIP,
        parameterization=geo.pr,
        fixed_dataset=True,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(internal_continuity, "internal_continuity")
    outlet_pressure = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.Outlet,
        parameterization=geo.pr,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(outlet_pressure, "outlet_pressure")
    no_slip = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.NoSlip,
        parameterization=geo.pr,
        criteria=Not(
            Or(
                geo.on_boundary_inlet,
                geo.on_boundary_outlet,
                geo.on_symmetry_xz,
                geo.on_symmetry_yz,
            )
        ),
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(no_slip, "no_slip")
    symmetry_xz = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        outvar={"v": 0},
        batch_size=cfg.batch_size.Symmetry // 2,
        parameterization=geo.pr,
        criteria=geo.on_symmetry_xz,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(symmetry_xz, "symmetry_xz")
    symmetry_yz = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        outvar={"u": 0},
        batch_size=cfg.batch_size.Symmetry // 2,
        parameterization=geo.pr,
        criteria=geo.on_symmetry_yz,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(symmetry_yz, "symmetry_yz")
    interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_z": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives=True,
        parameterization=geo.pr,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(interior, "interior")
    # 添加监视
    inlet_sample = geo.inlet.sample_boundary(1000, parameterization=geo.pr)
    outlet_sample = geo.outlet.sample_boundary(1000, parameterization=geo.pr)
    flow_domain.add_monitor(
        PointwiseMonitor(
            inlet_sample,
            ["p"],
            metrics={
                "inlet_pressure": lambda var: nd.dim(torch.mean(var["p"]), "kg/(m*s^2)")
            },
            nodes=flow_nodes,
        )
    )
    flow_domain.add_monitor(
        PointwiseMonitor(
            outlet_sample,
            ["p"],
            metrics={
                "outlet_pressure": lambda var: nd.dim(
                    torch.mean(var["p"]), "kg/(m*s^2)"
                )
            },
            nodes=flow_nodes,
        )
    )
    flow_domain.add_monitor(
        PointwiseMonitor(
            inlet_sample,
            ["u", "W_mi"],
            metrics={
                "inlet_u": lambda var: nd.dim(torch.mean(var["u"]), "m/s"),
                "inlet_vol_flow": lambda var: ureg.convert(
                    nd.dim(torch.mean(var["u"]) * torch.mean(var["W_mi"]), "m/s")
                    * nd.dim(manifold_inlet_height, "m")
                    * nd.dim(1, "m"),
                    "m^3/s",
                    "ml/s",
                ),
            },
            nodes=flow_nodes,
        )
    )
    flow_domain.add_monitor(
        PointwiseMonitor(
            outlet_sample,
            ["w"],
            metrics={"outlet_w": lambda var: nd.dim(torch.mean(var["w"]), "m/s")},
            nodes=flow_nodes,
        )
    )

    # 添加监控
    class SolverWithLog(Solver):
        def _cuda_graph_training_step(self, step: int):
            loss_static, losses_static = super()._cuda_graph_training_step(step)
            if step % cfg.training.print_stats_freq == 0:
                wandb.log({"losses": losses_static}, step=step, commit=False)
            return loss_static, losses_static

        def record_monitors(self, step: int):
            metrics = self.domain.rec_monitors(self.network_dir, self.writer, step)
            wandb.log({"metrics": metrics}, step=step, commit=True)
            return metrics

    # 求解
    flow_solver = SolverWithLog(cfg, flow_domain)
    flow_solver.solve()


if __name__ == "__main__":
    run()
