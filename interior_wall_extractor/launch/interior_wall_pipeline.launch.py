from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    namespace = 'interior_wall'

    return LaunchDescription([
        # Node(
        #     package='interior_wall_extractor',
        #     executable='preprocess_node',
        #     namespace=namespace,
        #     name='preprocess',
        #     parameters=[{
        #         'input_topic': '/synthetic_world/points',
        #         'output_topic': 'filtered_cloud',
        #         'process_service': 'process_cloud',
        #         'voxel_leaf_size': 0.05,
        #         'sor_mean_k': 30,
        #         'sor_stddev_mul': 1.0,
        #     }],
        # ),
        Node(
            package='interior_wall_extractor',
            executable='dbscan_node.py',
            namespace=namespace,
            name='dbscan',
            parameters=[{
                'input_topic': '/synthetic_world/points',
                'output_topic': 'clustered_cloud',
                'debug_rgb_topic': 'clustered_cloud_rgb',
                'eps': 0.09,
                'min_points': 10,
                'publish_largest_cluster_only': False,
                'publish_debug_rgb': True,
            }],
        ),
        # Node(
        #     package='interior_wall_extractor',
        #     executable='randla_bridge.py',
        #     namespace=namespace,
        #     name='randla',
        #     parameters=[{
        #         'input_topic': 'clustered_cloud',
        #         'output_topic': 'wall_clustered_cloud',
        #         'wall_semantic_label': 1,
        #         'checkpoint_path': '/absolute/path/to/randla_checkpoint.pth',
        #         'num_classes': 13,
        #         'publish_wall_only': True,
        #         'preserve_input_labels': True,
        #         'block_points': 4096,
        #         'tile_size': 4.0,
        #         'tile_overlap': 0.5,
        #         'score_threshold': 0.0,
        #         'feature_mode': 'xyz',
        #     }],
        # ),
        Node(
            package='interior_wall_extractor',
            executable='ransac_node',
            namespace=namespace,
            name='ransac',
            parameters=[{
                'input_topic': 'clustered_cloud',
                'output_topic': 'wall_planes_cloud',
                'debug_rgb_topic': 'wall_planes_cloud_rgb',
                'segment_per_input_label': True,
                'selected_room_id': 1,
                'plane_label_stride': 1000,
                'max_iterations': 500,
                'distance_threshold': 0.03,
                'min_plane_points': 100,
                'remaining_ratio_stop': 0.2,
                'publish_debug_rgb': True,
            }],
        ),
        # Node(
        #     package='interior_wall_extractor',
        #     executable='json_export_node',
        #     namespace=namespace,
        #     name='json_export',
        #     parameters=[{
        #         'input_topic': 'wall_planes_cloud',
        #         'output_directory': '/tmp/interior_wall_exports',
        #         'file_prefix': 'wall_planes',
        #         'export_on_receive': True,
        #     }],
        # ),
    ])
