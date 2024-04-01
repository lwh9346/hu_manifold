from hu_manifold_geometry import *
import torch
import modulus.sym
from matplotlib import pyplot as plt
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.hydra import ModulusConfig
from modulus.sym.key import Key


@modulus.sym.main(config_path="conf", config_name="conf_flow_fix")
def run(cfg: ModulusConfig) -> None:
    length_scale = quantity(cfg.custom.nd.length.value, cfg.custom.nd.length.unit)
    velocity_scale = quantity(cfg.custom.nd.velocity.value, cfg.custom.nd.velocity.unit)
    pressure_scale = quantity(cfg.custom.nd.pressure.value, cfg.custom.nd.pressure.unit)
    state_dict = torch.load(
        "/home/lwh/modulus-learn/outputs/hu_manifold_flow_fix/flow_network.0.pth"
    )
    input_keys = [Key("x"), Key("y"), Key("z")]
    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]
    flow_net = FourierNetArch(
        input_keys,
        output_keys,
        frequencies_params=[i for i in range(31)],
        skip_connections=True,
    )

    flow_net.load_state_dict(state_dict)
    res_x = 512
    res_z = 128
    x_start, x_end, z_start, z_end = (
        0.0,
        channel_length,
        microchannel_bottom_thickness,
        microchannel_bottom_thickness + microchannel_height + manifold_inlet_height,
    )
    x = torch.linspace(x_start, x_end, res_x)
    z = torch.linspace(z_start, z_end, res_z)
    xz_coords = torch.cartesian_prod(x, z)
    x_coords = xz_coords[:, 0:1]
    z_coords = xz_coords[:, 1:2]
    y_coords = torch.full((res_x*res_z, 1), SMM_width-0.01)
    uvwp = flow_net.forward({"x": x_coords, "y": y_coords, "z": z_coords})
    # print(uvwp)
    # u
    plt.clf()
    plt.figure(figsize=(15, 6))
    plt.tricontourf(
        x_coords.squeeze(),
        z_coords.squeeze(),
        uvwp["u"].detach().squeeze().numpy(),
        levels=80,
        cmap="rainbow",
        vmax=0.6,
        vmin=-0.6,
    )
    plt.colorbar()
    plt.savefig("outputu.png")
    # v
    plt.clf()
    plt.figure(figsize=(15, 6))
    plt.tricontourf(
        x_coords.squeeze(),
        z_coords.squeeze(),
        uvwp["v"].detach().squeeze().numpy(),
        levels=80,
        cmap="rainbow",
        vmax=0.6,
        vmin=-0.6,
    )
    plt.colorbar()
    plt.savefig("outputv.png")
    # w
    plt.clf()
    plt.figure(figsize=(15, 6))
    plt.tricontourf(
        x_coords.squeeze(),
        z_coords.squeeze(),
        uvwp["w"].detach().squeeze().numpy(),
        levels=80,
        cmap="rainbow",
        vmax=0.6,
        vmin=-0.6,
    )
    plt.colorbar()
    plt.savefig("outputw.png")
    pass


if __name__ == "__main__":
    run()
