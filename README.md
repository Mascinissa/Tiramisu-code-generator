# Automatic code generator for Tiramisu

*Forked form https://github.com/IsraMekki/tiramisu_code_generator *

Generates random Tiramisu codes. This version generates :
* codes with multiple computations, computations are randomly assigned to loops, different loops can share the same parent loop and computations can share the same loop level with another loop, the constraint is that top level loop must be shared across all computations (there can only be one root loop), interchange and tililing are applied only to the shared loops and unrolling is applied against the innermost loop 

# Running the generator
## Generator parameters
```
cd cmake-build-debug

#edit the "inputs.txt" file as needed
```
## Creating random samples
```
cd cmake-build-debug

make
./restructured
```
# Running the programs on Lanka
## Compiling all programs
```
#Programs to be executed are stored in /data/scratch/mmerouani/data/programs
cd /data/scratch/mmerouani/tiramisu
screen -S screenName                                                   #it's better to use screen

srun -N 1 -n 1 --exclusive -p lanka-v3 --pty bash -i
source ../anaconda3/bin/activate                          #activate the anaconda virtual environment
python3 compile.py --tiramisu-root $(pwd) --data-path ../data/ compile_all
```
# Running all programs
```
python3 compile.py --tiramisu-root $(pwd) --data-path ../data/ execute_all --timeout=150
```
