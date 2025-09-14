# Build Latency Prediction Models

```shell
conda activate voltanallm
cd prefill # or decode

# start prefill/decode server
./p_server.sh
# or ./p_server.sh <idx>, <idx> is GPU index
# or ./d_server.sh
# or ./d_server.sh <idx>

# run profile
# you MUST have sudo permission to set GPU frequency
python run_profile.py
# waiting to finish, for prefill, it should finish in ~10 min, for decode, it may consume several hours.
```

Scripts are for A100 80G GPU. You may need to adjust `--mem-fraction-static` (in `p_server.sh`/`d_server.sh`) and `MAX_TOKENS` (in `run_profile.py`) to fit your GPU memory.

You can select the model to profile by modifying `p_server.sh`/`d_server.sh`/`run_profile.py`.

After profiling, open and run `draw.ipynb` for result visualization and prediction model regression.
