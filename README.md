
## Download the raw data
Download the file GSE211692_RAW.tar: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211692


## Run the preprocessing
adjust the path and run
```
./preprocess.sh '-1' '/home/#####/cladcon_model/GSE211692_RAW.tar' '/home//#####//cladcon_model/preprocessed_tissue' '/home//#####//cladcon_model/workingdir_t' '/home//#####//cladcon_model/mapping_file.txt' '0' 'tissue'
./preprocess.sh '-1' '/home//#####//cladcon_model/GSE211692_RAW.tar' '/home//#####//cladcon_model/preprocessed_comb' '/home//#####//cladcon_model/workingdir_c' '/home//#####//cladcon_model/mapping_file.txt' '0' 'comb'
./preprocess.sh '-1' '/home//#####//cladcon_model/GSE211692_RAW.tar' '/home//#####//cladcon_model/preprocessed_full' '/home//#####//cladcon_model/workingdir_f' '/home//#####//cladcon_model/mapping_file.txt' '0' 'full'

```
 or use the SLURM submission file
 ```
 sbatch submit_preprocess.sub
 ```

## Run CLADCON
'tissue' for the tissue-of-origin task, 'comb' for the cancer task and 'full' for the disease task

```
python3 cladcon.py --task 'tissue'
python3 cladcon.py --task 'comb'
python3 cladcon.py --task 'full'
```
 or use the SLURM submission file
 ```
 sbatch submit_cladcon.sub
 ```

