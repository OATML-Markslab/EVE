import argparse
import os
import tqdm
import pickle
import pandas as pd
from utils import plot_helpers


def main(args):
    model_location = args.gmm_model_location
    with open(model_location, 'rb') as fid:
        gmm_model = pickle.load(fid)
    scores_location = args.input_eve_scores_location + os.sep + \
        'all_EVE_scores_'+ \
        args.output_eve_scores_filename_suffix+'.csv'
    all_scores = pd.read_csv(
        scores_location, 
        index=False
        )
    protein_list = list(all_scores.protein_name.unique())

    if args.plot_elbo_histograms:
        # Plot fit of mixture model to predicted scores
        histograms_location = \
            args.plot_location+os.sep+\
            'plots_histograms'+os.sep+\
            args.output_eve_scores_filename_suffix
        if not os.path.exists(histograms_location):
            os.makedirs(histograms_location)
        plot_helpers.plot_histograms(
            all_scores, 
            gmm_model,
            histograms_location,
            protein_list
        )

    if args.plot_scores_vs_labels:
        labels_dataset = pd.read_csv(
            args.labels_file_location,
            low_memory=False,
            usecols=['protein_name','mutations','ClinVar_labels']
        )
        labels_dataset = labels_dataset[labels_dataset.ClinVar_labels.isin([0,1])]
        all_scores_labelled = pd.merge(
            all_scores, 
            labels_dataset, 
            how='inner', 
            on=['protein_name','mutations']
        )
        labelled_scores_location = args.input_eve_scores_location + os.sep + \
        'all_EVE_scores_labelled_'+ \
        args.output_eve_scores_filename_suffix+'.csv'
        all_scores_labelled.to_csv(
            labelled_scores_location, index=False
        )

        # Plot scores against clinical labels
        scores_vs_labels_plot_location = \
            args.plot_location+os.sep+\
            'plots_scores_vs_labels'+os.sep+\
            args.output_eve_scores_filename_suffix
        if not os.path.exists(scores_vs_labels_plot_location):
            os.makedirs(scores_vs_labels_plot_location)
        for protein in tqdm.tqdm(protein_list,"Plot scores Vs labels"):
            protein_scores = all_scores_labelled[
                all_scores_labelled.protein_name==protein
            ]
            output_suffix = args.output_eve_scores_filename_suffix+'_'+protein
            plot_helpers.plot_scores_vs_labels(
                score_df=protein_scores, 
                plot_location=scores_vs_labels_plot_location,
                output_eve_scores_filename_suffix=output_suffix,
                mutation_name='mutations', 
                score_name="EVE_scores", 
                label_name='ClinVar_labels'
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot EVE scores against standard labels')
    parser.add_argument('--input_eve_scores_location', type=str, help='Folder where all EVE scores are stored')
    parser.add_argument('--input_eve_scores_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')
    parser.add_argument('--labels_file_location', default=None, type=str, help='File with ground truth labels for all proteins of interest (e.g., ClinVar)')
    parser.add_argument('--plot_location', default=None, type=str, help='Location of the different plots')
    parser.add_argument('--plot_elbo_histograms', default=False, action='store_true', help='Plots all evol indices histograms with GMM fits')
    parser.add_argument('--plot_scores_vs_labels', default=False, action='store_true', help='Plots EVE scores Vs labels at each protein position')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during run')
    args = parser.parse_args()


    main(args)