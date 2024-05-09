import os
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from sympy import Symbol, Eq, tanh, Or, And, Not, Gt, Lt, Abs
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
from modulus.sym.eq.pdes.navier_stokes import NavierStokes, Curl
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym import ureg, quantity
from modulus.sym.eq.non_dim import NonDimensionalizer

from u_manifold_geometry import MU

import wandb


@modulus.sym.main(config_path="conf", config_name="conf_u_flow_param_no_symmetry")
def run(cfg: ModulusConfig) -> None:
    # 初始化日志
    wandb_config = {
        "nondimensionalization": cfg.custom.nd,
        "batch_size": cfg.batch_size,
        "navier_stocks": cfg.custom.ns,
        "optimizer": cfg.optimizer,
    }
    wandb.init(project="u_manifold", config=wandb_config, save_code=True)
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
    input_keys = [
        Key("x"),
        Key("y"),
        Key("z"),
        Key("W_c"),
        Key("H_c"),
        Key("W_mi"),
        Key("W_mo"),
    ]
    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]
    flow_net = FullyConnectedArch(input_keys, output_keys)
    flow_nodes = (
        ns_nodes
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )
    geo = MU(nd, parameterized=True)
    # 边界条件
    flow_domain = Domain()
    inlet_vol_flow = quantity(
        cfg.custom.bc.inlet_vol_flow.value, cfg.custom.bc.inlet_vol_flow.unit
    )
    # 入口流量，要除以2，且因为法向量朝外，要加负号
    inlet_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        criteria=geo.on_boundary_inlet,
        outvar={"normal_dot_vel": -nd.ndim(inlet_vol_flow) / 2},
        batch_size=5,
        integral_batch_size=cfg.batch_size.Inlet,
        parameterization=geo.pr,
        fixed_dataset=True,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(inlet_continuity, "inlet_continuity")
    # 出口流量，要除以2
    outlet_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        criteria=geo.on_boundary_outlet,
        outvar={"normal_dot_vel": nd.ndim(inlet_vol_flow) / 2},
        batch_size=5,
        integral_batch_size=cfg.batch_size.Outlet,
        parameterization=geo.pr,
        fixed_dataset=True,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(outlet_continuity, "outlet_continuity")
    # 横向流量
    internal_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.mcip,
        outvar={"normal_dot_vel": nd.ndim(inlet_vol_flow) / 2},
        batch_size=5,
        integral_batch_size=cfg.batch_size.MCIP,
        parameterization=geo.pr,
        fixed_dataset=True,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(internal_continuity, "internal_continuity")
    outlet_pressure = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        criteria=geo.on_boundary_outlet,
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
        criteria=Not(geo.on_boundary_top),
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(no_slip, "no_slip")
    interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.fluid,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_z": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf") * 100,
            "momentum_x": Symbol("sdf") * 100,
            "momentum_y": Symbol("sdf") * 100,
            "momentum_z": Symbol("sdf") * 100,
        },
        compute_sdf_derivatives=True,
        parameterization=geo.pr,
        batch_per_epoch=cfg.custom.batch_per_epoch,
    )
    flow_domain.add_constraint(interior, "interior")
    # 添加监视
    inlet_sample = geo.fluid.sample_boundary(
        200, parameterization=geo.pr, criteria=geo.on_boundary_inlet
    )
    outlet_sample = geo.fluid.sample_boundary(
        200, parameterization=geo.pr, criteria=geo.on_boundary_outlet
    )
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
            ["normal_dot_vel", "W_mi"],
            metrics={
                "inlet_vol_flow": lambda var: nd.dim(
                    torch.mean(
                        var["normal_dot_vel"] * var["W_mi"] / 2 * geo.total_width_S
                    ),
                    "ul/s",
                ),
            },
            nodes=flow_nodes,
        )
    )
    flow_domain.add_monitor(
        PointwiseMonitor(
            outlet_sample,
            ["normal_dot_vel", "W_mo"],
            metrics={
                "outlet_vol_flow": lambda var: nd.dim(
                    torch.mean(
                        var["normal_dot_vel"] * var["W_mo"] / 2 * geo.total_width_S
                    ),
                    "ul/s",
                ),
            },
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
