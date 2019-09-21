import mido
import numpy
import torch
import numpy as np

def file_to_dataset(filename, program_type, split_width):
    midi = mido.MidiFile(filename)
    
    gm_melodic = list(range(0, 96)) + list(range(104, 112))
    gm_non_melodic = list(range(96, 104)) + list(range(112, 128))

    channels = set()

    for msg in midi:
        if msg.type == "program_change":
            if program_type == "melodic" and msg.program in gm_melodic:
                channels.add(msg.channel)
            if program_type == "non-melodic" and msg.program in gm_non_melodic:
                channels.add(msg.channel)

    # notes for each channel (key: channel id, value: tuple of (melody, velocity, length))
    notes = {}

    # elapsed times from the last note for each channel
    channel_elapsed_times = {}

    # current note
    channel_current_note = {}

    def add_elapsed_time(t):
        for ch in channel_elapsed_times:
            channel_elapsed_times[ch] += t

    for ch in channels:
        notes[ch] = []
        channel_elapsed_times[ch] = 0
        channel_current_note[ch] = (None, None)

    for msg in midi:
        add_elapsed_time(msg.time)
        if msg.type == "note_on" and msg.channel in channels:
            # add last note
            notes[msg.channel].append((
                channel_current_note[msg.channel][0],
                channel_current_note[msg.channel][1],
                channel_elapsed_times[msg.channel]
            ))
            
            # reset counter
            channel_elapsed_times[msg.channel] = 0
            
            # update current note
            channel_current_note[msg.channel] = (msg.note, msg.velocity)
        if msg.type == "note_off" and msg.channel in channels:
            # add last note
            notes[msg.channel].append((
                channel_current_note[msg.channel][0],
                channel_current_note[msg.channel][1],
                channel_elapsed_times[msg.channel]
            ))

            # reset counter
            channel_elapsed_times[msg.channel] = 0

            # update current note
            channel_current_note[msg.channel] = (0, 0)

    # sanitize output
    for ch in channels:
        notes[ch] = list(filter(lambda x: x[0] != None and x[2] != 0, notes[ch]))

    # re-format notes
    notes_reform = []
    for i in notes:
        notes_reform.append([])
        current = notes_reform[-1]
        for note in notes[i]:
            if note[1] == 0:
                melody = 0
            else:
                melody = note[0] + 1
            time = int(note[2] * 100)
            current.append([melody, time])

    # split notes into arrays of length of split_width, changing the offset one by one
    notes_split = []
    for lst in notes_reform:
        notes_split += [lst[i : i + split_width] for i in range(0, len(lst) - split_width + 1)]

    return notes_split

import torch.utils.data.dataset
import os

class MidiDataset(torch.utils.data.Dataset):
    # dir: root dir of dataset
    # program_type: 'melodic' or 'non-melodic'. specifies type of instrument
    # split_width: specifies length of each data
    # transform: post process transform function callback
    def __init__(self, dir, program_type, split_width, transform=None):
        self.dir = dir
        self.program_type = program_type
        self.split_width = split_width
        self.filelist = list(filter(lambda x: x.endswith('.mid'), os.listdir(dir)))
        self.transform = transform
        self.datalist = []

        for file in self.filelist:
            self.datalist += file_to_dataset(os.path.join(dir, file), self.program_type, self.split_width)

        self.datalist = torch.tensor(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]

    def __len__(self):
        return len(self.datalist)
