export MSA_data_folder='./data/MSA'
export MSA_list='./data/mappings/example_mapping.csv'
export MSA_weights_location='./data/weights'
export VAE_checkpoint_location='./results/VAE_parameters'
export model_name_suffix='Jan1_PTEN_example'
export model_parameters_location='./EVE/default_model_params.json'
export training_logs_location='./logs/'
export protein_index=0

python train_VAE.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} \
    --verbose

export computation_mode='all_singles'
export all_singles_mutations_folder='./data/mutations'
export evol_indices_location='./results/evol_indices'
export num_samples_compute_evol_indices=20000
export batch_size=2048

python predict_ELBO.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --computation_mode ${computation_mode} \
    --all_singles_mutations_folder ${all_singles_mutations_folder} \
    --output_evol_indices_location ${output_evol_indices_location} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size}
    --verbose

export evol_indices_filename_suffix='_20000_samples'
export protein_list='./data/mappings/example_mapping.csv'
export eve_scores_location='./results/EVE_scores'
export eve_scores_filename_suffix='Jan1_PTEN_example'
export GMM_parameter_location='./results/GMM_parameters/Default_GMM_parameters'
export GMM_parameter_filename_suffix='default'
export protein_GMM_weight=0.3
export default_uncertainty_threshold_file_location='./utils/default_uncertainty_threshold.json'

python predict_GMM_score.py \
    --input_evol_indices_location ${evol_indices_location} \
    --input_evol_indices_filename_suffix ${evol_indices_filename_suffix} \
    --protein_list ${protein_list} \
    --output_eve_scores_location ${eve_scores_location} \
    --output_eve_scores_filename_suffix ${eve_scores_filename_suffix} \
    --GMM_parameter_location ${GMM_parameter_location} \
    --GMM_parameter_filename_suffix ${GMM_parameter_filename_suffix} \
    --protein_GMM_weight ${protein_GMM_weight} \
    --compute_uncertainty_thresholds \
    --verbose

export plot_location='./results'
export labels_file_location='./data/labels/PTEN_ClinVar_labels.csv'

python plot_scores_and_labels.py \
    --input_eve_scores_location ${eve_scores_location} \
    --input_eve_scores_filename_suffix ${eve_scores_filename_suffix} \
    --labels_file_location ${labels_file_location} \
    --plot_location ${plot_location} \
    --plot_histograms \
    --plot_scores_vs_labels \
    --verbose