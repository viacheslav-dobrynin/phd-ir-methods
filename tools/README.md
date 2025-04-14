# PyLucene Install
1. Create or activate conda environment:
```shell
conda create -n pylucene python=3.10
conda activate pylucene
```
2. Unzip the archive with PyLucene: 
```shell
tar -xzf pylucene.tar.gz
```
3. Unzip the archive with JCC: 
```shell
tar -xzf jcc.tar.gz
```
4. Copy `libjcc3.so` to LD_LIBRARY_PATH. 
`libjcc3.so` is located in `jcc` folder from previous step.
```shell
cp jcc/libjcc3.so "$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
```
5. Add existing `libjvm.so` to LD_LIBRARY_PATH.
`libjvm.so` is located in Java folder (it should be 21 version or higher).
```shell
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/path/to/java/lib/server/"
```
6. Add `jcc` folder to `sys.path`:
```shell
import sys

sys.path.append('/path/to/jcc')
```
7. Check that everything works:
```shell
import sys
sys.path.append('/path/to/jcc')

import lucene

lucene.initVM()
print("Done")
```
