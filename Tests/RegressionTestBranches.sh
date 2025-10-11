#!/bin/zsh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate BabelBrain #activating 1st reference 
git checkout main
sed -i '' "/gen_output_folder/c\\
gen_output_folder = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.4.3
" Tests/config.ini
pytest -k "test_generate_valid_outputs and Metal and ((H317 and 250) or CTX_500) and None" -m "basic_babelbrain_params"
conda deactivate
conda activate BabelMLX #activating 2nd reference 
git checkout MLXv2
#setting output directory
sed -i '' "/gen_output_folder/c\\
gen_output_folder = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.8.0
" Tests/config.ini
pytest -k "test_generate_valid_outputs and MLX and ((H317 and 250) or CTX_500) and None" -m "basic_babelbrain_params"
#setting comparison directories
sed -i '' "/ref_output_folder_1/c\\
ref_output_folder_1 = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.4.3
" Tests/config.ini
sed -i '' "/ref_output_folder_2/c\\
ref_output_folder_2 = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.8.0
" Tests/config.ini
# run comparison test
pytest -k "test_full_pipeline_two_outputs"
conda deactivate


