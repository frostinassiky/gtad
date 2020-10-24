import os
import math
import numpy as np
import pandas as pd
import torch.nn.parallel

from gtad_lib import opts
from gtad_lib.models import GTAD
from gtad_lib.dataset import VideoDataSet

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt['output'] + "/results"):
        os.makedirs(opt['output'] + "/results")

    model = GTAD(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load(opt["output"] + "/GTAD_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", mode='inference'),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    print("Inference start")
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            # offset = min(test_loader.dataset.data['indices'][idx[0]])
            # video_name = video_name+'_{}'.format(math.floor(offset/250))
            input_data = input_data.cuda()

            # forward pass
            confidence_map, start, end = model(input_data)

            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            max_start = max(start_scores)
            max_end = max(end_scores)

            # use BMN post-processing to boost performance
            start_bins = np.zeros(len(start_scores))
            start_bins[0] = 1  # [1,0,0...,0,1]
            for idx in range(1, tscale - 1):
                if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                    start_bins[idx] = 1
                elif start_scores[idx] > (0.5 * max_start):
                    start_bins[idx] = 1

            end_bins = np.zeros(len(end_scores))
            end_bins[-1] = 1
            for idx in range(1, tscale - 1):
                if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                    end_bins[idx] = 1
                elif end_scores[idx] > (0.5 * max_end):
                    end_bins[idx] = 1

            # enumerate sub-graphs as proposals
            new_props = []
            for idx in range(opt["max_duration"]):
                for jdx in range(opt["temporal_scale"]):
                    start_index = jdx
                    end_index = start_index + idx+1
                    if end_index < opt["temporal_scale"] and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                        xmin = start_index / opt['temporal_scale']
                        xmax = end_index / opt['temporal_scale']
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        new_props.append([xmin, xmax, clr_score, reg_score])
            new_props = np.stack(new_props)

            col_name = ["xmin", "xmax", "clr_score", "reg_socre"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv(opt["output"]+"/results/" + video_name + ".csv", index=False)

    print("Inference finished")
