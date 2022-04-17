animal
======

Yet another RL library with a weird name. Made for the final project for the GATech OMSCS Deep Learning class.

## Install

Clone the repo, `cd` into it, and use `pip` to install.

```bash
git clone https://github.com/jfpettit/kindling.git
cd kindling
pip install -e .
```

You should be set.

## Running

Run an algorithm by invoking `python` and providing the filename for that algorithm as an argument:

```bash
python ppo.py
```

Add the `--help` flag to the call to see argument options, e.g. (`python ppo.py --help`).


## Guarantees

There are none, except that so far this code has been tested on Mac and can be expected to work on UNIX/Linux-like systems. Windows needs to be tested.
