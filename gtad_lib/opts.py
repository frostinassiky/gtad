import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.004)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=10)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32)
    parser.add_argument(
        '--step_size',
        type=int,
        default=5)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    parser.add_argument(
        '--n_gpu',
        type=int,
        default=6)
    # output settings
    parser.add_argument('--eval', type=str, default='validation')
    parser.add_argument('--output', type=str, default="./output")
    # Overall Dataset settings
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/hacs_annotations/HACS_segments_v1.1.1.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="./data/hacs_feature_mit/csv_mean_100.hdf5")

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048) 
    parser.add_argument(
        '--h_dim_1d',
        type=int,
        default=256) 
    parser.add_argument(
        '--h_dim_2d',
        type=int,
        default=128) 
    parser.add_argument(
        '--h_dim_3d',
        type=int,
        default=512) 

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.8)
    parser.add_argument(
        '--result_file',
        type=str,
        default="result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="evaluation_result.pdf")

    # anchors
    parser.add_argument('--max_duration', type=int, default=100)  # anet: 100 snippets
    parser.add_argument('--min_duration', type=int, default=0)  # anet: 100 snippets


    # ablation settings

    parser.add_argument(
        '--goi_samp',
        type=int,
        default=0) # 0: sample all frame; 1: sample each output position
    parser.add_argument(
        '--goi_style',
        type=int,
        default=1)  # 0: no context, 1: last layer context, 2: all layer context
    parser.add_argument(
        '--kern_2d',
        type=int,
        default=1) # 3
    parser.add_argument(
        '--pad_2d',
        type=int,
        default=0) # 1

    args = parser.parse_args()

    return args

