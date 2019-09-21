import mido

# input: array of tuple(melody, length), output: midi object
def notes_to_mid(notes):
    midi = mido.MidiFile()
    midi.ticks_per_beat = 100
    meta_track = mido.MidiTrack()
    midi.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage("track_name", name="generated"))
    meta_track.append(mido.MetaMessage("track_name", name="Conductor Track"))
    meta_track.append(mido.MetaMessage("marker", text="Setup"))
    meta_track.append(mido.MetaMessage("key_signature", key="C"))
    # meta_track.append(mido.MetaMessage("time_signature", numerator=1, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8))
    meta_track.append(mido.MetaMessage("set_tempo", tempo=1000000))
    # meta_track.append(mido.MetaMessage("end_of_track", time=?????))
    
    main_track = mido.MidiTrack()
    midi.tracks.append(main_track)
    main_track.append(mido.MetaMessage("track_name", name="main"))
    main_track.append(mido.Message("control_change", channel=0, control=121, value=0))
    main_track.append(mido.Message("control_change", channel=0, control=7, value=100))
    main_track.append(mido.Message("control_change", channel=0, control=10, value=64))
    main_track.append(mido.Message("control_change", channel=0, control=0, value=0))
    main_track.append(mido.Message("control_change", channel=0, control=32, value=0))
    main_track.append(mido.Message("program_change", channel=0, program=0))
    
    for (melody, length) in notes:
        if melody == 0:
            main_track.append(mido.Message("note_on", channel=0, note=0, velocity=0, time=length))
        else:
            main_track.append(mido.Message("note_on", channel=0, note=melody + 1, velocity=100, time=0))
            main_track.append(mido.Message("note_on", channel=0, note=melody + 1, velocity=0, time=length))

    return midi