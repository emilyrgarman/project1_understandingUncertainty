# %%
import pandas as pd
import ast
import re

# note: you may need to install fsspec, huggingface_hub to load the data in
df = pd.read_csv("hf://datasets/ailsntua/Chordonomicon/chordonomicon_v2.csv")

# filter Chordonomicon dataset for Taylor Swift -- id taken from her Spotify homepage
swift = df.query("spotify_artist_id == '06HL4z0CvFAxyc27GXpf02'")

# read the mapping CSV file in -- downloaded from https://github.com/spyroskantarelis/chordonomicon/blob/main/chords_mapping.csv
chord_relations = pd.read_csv('chords_mapping.csv')

# create a dictionary with keys the "chords" and values the "degrees" -- code from https://github.com/spyroskantarelis/chordonomicon/blob/main/convert_to_mappings.ipynb
# note: we don't actually use chord_notes (e.g., do, re, mi) later on, and are more so focused on the 12-semitone binary representation found in chord_degrees
chord_degrees = dict(zip(chord_relations['Chords'], chord_relations['Degrees']))
for key, value in chord_degrees.items():
    chord_degrees[key] = ast.literal_eval(value)
chord_notes = dict(zip(chord_relations['Chords'], chord_relations['Notes']))
for key, value in chord_notes.items():
    chord_notes[key] = ast.literal_eval(value)

# %%
### CLEAN THE CHORD DATA

# remove all structural labels (e.g., <verse_1>, <intro_1>, <chorus_1>)
# finds a <, followed by characters contained within brackets, followed by >
swift['chords'] = swift['chords'].apply(lambda s: re.sub(r"<[^>]*>", "", s))

# remove chord inversions/bass notes (e.g., /G, /C, /B)
# finds a slash followed by any non-space or non-tab characters until the end of a chord
swift['chords'] = swift['chords'].apply(lambda s: re.sub(r"/[^ \t]*", "", s))

# convert long string of chords into list of individual chord strings
swift['chords'] = swift['chords'].apply(lambda s: s.split())

# %%
### EXPORT CHORD DATA -- formatted as list of individual chords, such that each row represents the chord progression for an individual song
# note: the way that the data is formatted here allows us to do what we did for the taxi data, but it presumably needs to be formatted differently for music21
swift['chords'].to_csv('swift_chords.csv', index = True)

# %%
### MAPPING CHORDS TO DEGREES

# explode the chord data such that each chord gets its own row
swift_exploded = swift.explode('chords').reset_index(drop = True)
swift_exploded.rename(columns = {'chords': 'chord'}, inplace = True)

# time refers to index for each chord within its song
swift_exploded['time'] = swift_exploded.groupby('id').cumcount() + 1

# root refers to root of the chord
swift_exploded['root'] = swift_exploded['chord'].str[0]

# degrees maps the chord column to the chord_degrees dictionary
swift_exploded['degrees'] = swift_exploded['chord'].map(chord_degrees)

# converts list in the degrees column to a dataframe, to create our 1-12 column semitone binary representation   
swift_degrees = swift_exploded['degrees'].apply(pd.Series)
cols = {i: i + 1 for i in range(12)}
swift_degrees.rename(columns = cols, inplace = True)

# concatenate new columns back to swift_exploded to create our final mapped chord data
swift_mapped = pd.concat([swift_exploded[['id', 'time', 'chord', 'root']], swift_degrees], axis = 1)

# %%
### EXPORT MAPPED CHORDS -- formatted as each row representing an individual chord within a song, with their degree mapping
# note: this format matches bach.data -- should allow us to use music21
swift_mapped.to_csv('swift_mapped.csv', index = False)