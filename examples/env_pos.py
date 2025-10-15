import torch
import argparse

import genesis as gs

NUM_ENVS = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-6.0, 0.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.35),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=True,
        )
    )

    scene.build(n_envs=NUM_ENVS)

    pos = torch.zeros((NUM_ENVS, 3), device=gs.device, dtype=gs.tc_float)
    pos[0] = torch.tensor([0, 0, 0.35], device=gs.device, dtype=gs.tc_float)
    pos[1] = torch.tensor([2, 2, 0.35], device=gs.device, dtype=gs.tc_float)

    # Set the positions to two of the four environments
    envs_idx = [0, 1]
    robot.set_pos(pos[envs_idx], envs_idx=envs_idx)


if __name__ == "__main__":
    main()
