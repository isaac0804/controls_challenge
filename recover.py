import io
import pickle

from groundtruth_training_muzero import ReplayBuffer

b = ReplayBuffer()

with open("random_replay_buffer copy.pickle", "rb") as f:
    corrupted_data = io.BytesIO(f.read())
    # Use the pure-Python version, we can't see the internal state of the C version
    unpickler = pickle._Unpickler(corrupted_data)
    try:
        unpickler.load()
    except EOFError:
        pass

    metastack = unpickler.metastack
    mgr = metastack[1]
    b.buffer = mgr[3]

with open("random_replay_buffer copy.pickle", "wb") as f:
    pickle.dump(b, f)