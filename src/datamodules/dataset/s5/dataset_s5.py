import os
import json
import torch
import numpy as np
import random

from src.modules.spatialscaper2.semseg_spatialscaper2 import SemgSegScaper2
from src.utils import LABELS

def collate_fn(list_data_dict):
    data = {k: [] for k in list_data_dict[0].keys()}
    for ddict in list_data_dict:
        for k in data:
            data[k].append(ddict[k])
    for k in data.keys():
        if type(data[k][0]) is torch.Tensor:
            data[k] = torch.stack(data[k], 0)
    return data

class DatasetS5(torch.utils.data.Dataset):
    def __init__(self,
                 config, # dict or string
                 n_sources,
                 label_set, # 'dcase2025t4' key of LABELS in utils
                 return_dry=False, # for source separation
                 label_vector_mode='multihot', # multihot, concat, stack
                 checking=None, # return all the wet source, dry source, ir from spatial scaper
                 use_full=False,
                 use_additional_irs=False, # False -> use default splits; full_t -> add 2 more from FO-MEIR test; full_ts -> add all from FO-MEIR (test + reverb-S)
                 use_additional_backgrounds=False
                ):
        super().__init__()
        self.checking = checking
        self.label_set = label_set
        self.config = config
        self.n_sources = n_sources
        self.return_dry = return_dry
        self.label_vector_mode = label_vector_mode
        self.use_full = use_full

        if isinstance(config, str): # path to json
            self.from_metadata = True
            self.metadata_dir = os.path.dirname(config)
            # print("config: ", config)
            with open(config) as f:
                self.data = json.load(f)
            with open(os.path.join(self.metadata_dir, self.data[0]['metadata_path'])) as f:
                one_ss_config = json.load(f)
                self.sr = one_ss_config['config']['sr']
                self.duration = one_ss_config['config']['duration']
            self.dataset_length = len(self.data)
        else:
            self.from_metadata = False
            self.spatialscaper = config['spatialscaper']
            self.sr = config['spatialscaper']['sr'] if 'spatialscaper' in config else 32000
            self.duration = config['spatialscaper']['duration'] if 'spatialscaper' in config else 10.0
            self.snr_range = config['snr_range']
            self.nevent_range = config['nevent_range']
            self.dataset_length = config['dataset_length']
            self.shuffle_label = config['shuffle_label']

        self.labels = LABELS[self.label_set]
        self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
        self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
        self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False,  dtype=torch.float32)

        self.collate_fn = collate_fn


    def get_onehot(self, label):
        return self.label_onehots[label]

    def __len__(self):
        return self.dataset_length

    def get_sound_scape(self, idx):
        if self.from_metadata:  # Generate soundscape from json config
            metadata_path = os.path.join(self.metadata_dir, self.data[idx]['metadata_path'])
            ssc = SemgSegScaper2.from_metadata(metadata_path)
        else: # randomly generate sound scape from param config
            # initialize object
            ssc = SemgSegScaper2(**self.spatialscaper)

            # set room
            ssc.set_room(('choose', [])) # random

            # add events
            nevents = random.randint(self.nevent_range[0], self.nevent_range[1])  # nr of events
            for i in range(nevents):
                ssc.add_event(
                    label=("choose_wo_replacement", []),
                    source_file=("choose", []),
                    source_time=("choose", []),
                    event_time=None,
                    event_position=('choose', []),
                    snr=("uniform", self.config['snr_range'][0], self.config['snr_range'][1]),
                    split=None,
                )
            assert self.nevent_range[0] <= len(ssc.fg_events) <=self.nevent_range[1]

            # Add interference events
            if 'interference_dir' in self.config['spatialscaper']:
                ninteferences = random.randint(self.config['ninterference_range'][0], self.config['ninterference_range'][1])
                for _ in range(ninteferences):
                    ssc.add_interference(
                        label=("choose", []),
                        source_file=("choose", []),
                        source_time=("choose", []),
                        event_time=None,
                        event_position=('choose', []),
                        snr=("uniform", self.config['inteference_snr_range'][0], self.config['inteference_snr_range'][1]),
                        split=None,
                    )
            # add background, make sure it is consistent with room
            if self.spatialscaper['background_dir']: # only add noise if there is background_dir
                ssc.add_background(source_file = ('choose_wo_room_consistency', []))
        output = ssc.generate()
        assert(len(set(output['labels'])) == len(output['labels'])), 'duplicated sound events in the mixture'
        return output

    def __getitem__(self, idx): # override getitem
        output = self.get_sound_scape(idx)
        if len(output['labels']) < self.n_sources:
            for _ in range(self.n_sources - len(output['labels'])):
                output['labels'].append('silence')
                if self.return_dry:
                    output['dry_sources'].append(np.zeros_like(output['dry_sources'][0]))

        if self.from_metadata: # get predefined ref event
            even_order = self.data[idx]['ref_event']
            assert set(even_order) == set(output['labels'])
            indices_order = [output['labels'].index(label) for label in even_order]
        else:
            indices_order = list(range(len(output['labels'])))
            if self.shuffle_label:
                random.shuffle(indices_order)
            even_order = [output['labels'][i] for i in indices_order]

        label_vector_all = torch.stack([self.get_onehot(label) for label in output['labels']]) # [nevent, nclass]
        label_vector_all = label_vector_all[indices_order, ...] # change order of sound events
        if self.label_vector_mode == 'multihot': label_vector_all = torch.any(label_vector_all.bool(), dim=0).float() # [nclass]
        elif self.label_vector_mode == 'concat': label_vector_all = label_vector_all.flatten() # [nevent x nclass]
        elif self.label_vector_mode == 'stack': pass  # [nevent, nclass]
        else: raise NotImplementedError(f'label_vector_mode of "{self.label_vector_mode}" has not been implemented')

        # convert string labels to one-hot
        output['on_offset'] = {self.get_onehot(label): output['on_offset'][label] for label in output['on_offset']}

        item = {
            'mixture': torch.from_numpy(output['mixture'].transpose()).to(torch.float32), # [nch, wlen]
            'label_vector': label_vector_all, # [nevent, nclass], [nclass], or [nevent x nclass]
            'label': even_order, # list
            'on_offset': output['on_offset'],
            'sr': self.sr,
            'duration': self.duration,
        }

        if self.return_dry:
            dry_sources_all = np.stack([dry_source for dry_source in output['dry_sources']])[:, None, :]# [nevent, 1, wlen]
            dry_sources_all = dry_sources_all[indices_order, ...] # change order of sound events
            dry_sources = torch.from_numpy(dry_sources_all)
            if self.return_dry: item['dry_sources']= dry_sources.to(torch.float32) # [nsources, 1 ch, wlen]

        if self.checking: item['spatialscaper'] = output # return everything

        return item
