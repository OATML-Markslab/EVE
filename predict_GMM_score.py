import os
import numpy as np
import pandas as pd
import argparse
import pickle
import tqdm
import json
from utils import performance_helpers as ph, plot_helpers
from EVE import GMM_model


def preprocess_evol_indices(all_evol_indices, protein_name=None, verbose=False):
    evol_indices = all_evol_indices.drop_duplicates()
    X_train = evol_indices['evol_indices'].values.reshape(-1, 1)
    proteins_train = list(evol_indices['protein_name'].values)
    if verbose:
        print("Training data size: "+str(len(X_train)))
        print("Number of distinct proteins in protein_list: "+str(len(np.unique(all_evol_indices['protein_name']))))
    return X_train, proteins_train


def predict_scores_GMM(gmm_model, all_evol_indices, 
    recompute_uncertainty_threshold = True, uncertainty_threshold_location = None,
    verbose = False):
    
    all_preds = all_evol_indices.copy()

    if gmm_model.protein_GMM_weight > 0.0:
        all_preds['scores'] = np.nan
        all_preds['classes'] = ""
        protein_list = list(gmm_model.models.keys()).drop('main')
        for protein in tqdm.tqdm(protein_list,"Scoring all protein mutations"):
            preds_protein = all_preds[all_preds.protein_name==protein].copy()
            X_pred_protein = preds_protein['evol_indices'].values.reshape(-1, 1)
            scores, classes = gmm_model.predict_weighted(gmm_model, X_pred_protein)
            preds_protein['scores'] = scores
            preds_protein['classes'] = classes
            all_preds.loc[all_preds.protein_name==protein, :] = preds_protein

    else:
        X_pred = all_preds['evol_indices'].values.reshape(-1, 1)
        scores, classes = gmm_model.predict(X_pred, 'main')
        all_preds['scores'] = scores
        all_preds['classes'] = classes

    if verbose:
        scores_stats = all_preds['scores'].describe()
        print("Score stats: \n", scores_stats)
        len_before_drop_na = len(all_preds)
        len_after_drop_na = len(all_preds['scores'].dropna())
        print("Dropped mutations due to missing EVE scores: "+str(len_after_drop_na-len_before_drop_na))

    return all_preds


def filter_uncertainties(all_scores, n_quantiles, default_uc_location, recompute = False):
    # Compute uncertainty from mixture model
    y_pred = all_scores['scores']
    uncertainty = ph.predictive_entropy_binary_classifier(y_pred)
    all_scores['uncertainty'] = uncertainty

    # Get quantiles for uncertainty
    if not recompute:
        with open(default_uc_location,'r') as fid:
            uc_quantiles = json.load(fid)
    uc_quantiles = ph.get_uncertainty_thresholds(uncertainty, n_quantiles)
    if verbose:
        print('Quantiles', uc_quantiles)
    
    # Assign classes at each quantile
    for i, quantile in enumerate(quantiles):
        level = f'class_ucq_{i}'
        all_scores[level] = all_scores['class'] * (uncertainty < quantile)
        if verbose:
            print("Stats classification by uncertainty for quantile #:"+str(quantile))
            print(all_scores[level].value_counts(normalize=True))
    return all_scores


def main(args):
    # Load evolutionary indices from files
    mapping_file = pd.read_csv(args.protein_list,low_memory=False)
    protein_list = np.unique(mapping_file['protein_name'])
    list_variables_to_keep=['protein_name','mutations','evol_indices']
    all_evol_indices = []
    for protein in protein_list:
        evol_indices_location = \
            args.input_evol_indices_location + os.sep + \
            protein + args.input_evol_indices_filename_suffix + '.csv'
        if os.path.exists(evol_indices_location):
            evol_indices = pd.read_csv(
                evol_indices_location, 
                low_memory=False, ignore_index=True,
                usecols=list_variables_to_keep)
            all_evol_indices.append(evol_indices)
    all_evol_indices = pd.concat(all_evol_indices)
           
    if args.load_GMM_models:
        # Load GMM models from file
        gmm_model_location = \
            args.GMM_parameter_location+os.sep+\
            'GMM_model_dictionary_'+ \
            args.GMM_parameter_filename_suffix
        with open(gmm_model_location, "rb" ) as fid:
            gmm = pickle.load(fid)

    else:
        # train GMMs on mutation evolutionary indices
        gmm_params = {
            'protein_GMM_weight':args.protein_GMM_weight
            }
        gmm_model = GMM_model.GMM_model(gmm_params)
        gmm_model.fit(
            all_evol_indices, 
            protein_list
            )

        # Write GMM models to pickle file
        gmm_model_location = \
            args.GMM_parameter_location+os.sep+\
            'GMM_model_dictionary_'+ \
            args.output_eve_scores_filename_suffix
        with open(gmm_model_location, "wb" ) as fid:
            pickle.dump(gmm_model, fid)
    
    # Compute EVE classification scores for all mutations and write to file
    all_scores = predict_scores_GMM(
        gmm_model,
        all_evol_indices, 
        protein_list,
        args.recompute_uncertainty_threshold
    )
    all_scores.to_csv(
        args.output_eve_scores_location+os.sep+ \
        'all_EVE_scores_'+ \
        args.output_eve_scores_filename_suffix+'.csv', 
        index=False
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GMM fit and EVE scores computation')
    parser.add_argument('--input_evol_indices_location', type=str, help='Folder where all individual files with evolutionary indices are stored')
    parser.add_argument('--input_evol_indices_filename_suffix', type=str, default='', help='Suffix that was added when generating the evol indices files')
    parser.add_argument('--protein_list', type=str, help='List of proteins to be included (one per row)')
    parser.add_argument('--output_eve_scores_location', type=str, help='Folder where all EVE scores are stored')
    parser.add_argument('--output_eve_scores_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')

    parser.add_argument('--load_GMM_models', default=False, action='store_true', help='If True, load GMM model parameters. If False, train GMMs from evol indices files')
    parser.add_argument('--GMM_parameter_location', default=None, type=str, help='Folder where GMM objects are stored if loading / to be stored if we are re-training')
    parser.add_argument('--GMM_parameter_filename_suffix', default=None, type=str, help='Suffix of GMMs model files to load')
    parser.add_argument('--protein_GMM_weight', default=0.3, type=float, help='Value of global-local GMM mixing parameter')

    parser.add_argument('--compute_EVE_scores', default=False, action='store_true', help='Computes EVE scores and uncertainty metrics for all input protein mutations')
    parser.add_argument('--recompute_uncertainty_threshold', default=False, action='store_true', help='Recompute uncertainty thresholds based on all evol indices in file. Otherwise loads default threhold.')
    parser.add_argument('--default_uncertainty_threshold_file_location', default='./utils/default_uncertainty_threshold.json', type=str, help='Location of default uncertainty threholds.')

    parser.add_argument('--plot_histograms', default=False, action='store_true', help='Plots all evol indices histograms with GMM fits')
    parser.add_argument('--plot_scores_vs_labels', default=False, action='store_true', help='Plots EVE scores Vs labels at each protein position')
    parser.add_argument('--labels_file_location', default=None, type=str, help='File with ground truth labels for all proteins of interest (e.g., ClinVar)')
    parser.add_argument('--plot_location', default=None, type=str, help='Location of the different plots')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during run')
    args = parser.parse_args()
    
    main(args)
    