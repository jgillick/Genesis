import argparse
import numpy as np

import genesis as gs
from genesis.sensors.raycaster.patterns import DepthCameraPattern, GridPattern, SphericalPattern
from genesis.utils.geom import euler_to_quat

MOVE_RADIUS = 1.0
MOVE_RATE = 1.0 / 100.0

def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
    parser.add_argument("-B", "--n_envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument(
        "--pattern", type=str, default="spherical", choices=("spherical", "depth", "grid", "none"), help="Sensor pattern type"
    )
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

    sensor_kwargs = dict(
        entity_idx=robot.idx,
        pos_offset=(0.3, 0.0, 0.1),
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
        draw_debug=True,
    )

    if args.pattern == "depth":
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=DepthCameraPattern(), **sensor_kwargs))
        scene.start_recording(
            data_func=(lambda: sensor.read_image()[0]) if args.n_envs > 0 else sensor.read_image,
            rec_options=gs.recorders.MPLImagePlot(),
        )
    elif args.pattern == "grid":
        sensor_kwargs['pos_offset'] = (0.0, 0.0, 0.1)
        sensor = scene.add_sensor(
            gs.sensors.Lidar(
                pattern=GridPattern(resolution=0.2, size=(0.4, 0.2)), 
                **sensor_kwargs
            )
            )
    elif args.pattern == "spherical":
        sensor = scene.add_sensor(gs.sensors.Lidar(pattern=SphericalPattern(), **sensor_kwargs))

    scene.build(n_envs=args.n_envs)

    

    init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    target_pos = init_pos.copy()
    target_euler = init_euler.copy()

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        if args.n_envs > 0:
            pos_np = np.expand_dims(pos_np, axis=0).repeat(args.n_envs, axis=0)
            quat_np = np.expand_dims(quat_np, axis=0).repeat(args.n_envs, axis=0)
        robot.set_pos(pos_np)
        robot.set_quat(quat_np)

    apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

    try:
        while True:
            scene.step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
