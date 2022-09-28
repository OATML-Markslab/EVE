import os,sys
import json
import argparse
import pandas as pd
import torch

from EVE import VAE_model
from utils import data_utils

def main(args):
    # Load protein name
    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    print("Protein name: "+str(protein_name))
    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    # Load and preprocess MSA
    msa_location = \
        args.MSA_data_folder + os.sep + \
        mapping_file['msa_location'][args.protein_index]
    print("MSA file: "+str(msa_location))
    weights_location = \
        args.MSA_weights_location + os.sep + \
        protein_name + '_theta_' + str(theta) + '.npy'
    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            weights_location=weights_location,
            theta=theta,
            use_weights=True
    )

    # Load model
    model_name = protein_name + "_" + args.model_name_suffix
    print("Model name: "+str(model_name))
    model_params = json.load(open(args.model_parameters_location))
    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42
    )
    model = model.to(model.device)
    try:
        checkpoint_name = \
            str(args.VAE_checkpoint_location) + os.sep + \
            model_name + "_final"
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
    except:
        print("Unable to locate VAE model checkpoint")
        sys.exit(0)


    # Load mutations location
    if args.computation_mode=="all_singles":
        mutations_location = \
            args.all_singles_mutations_folder + os.sep + \
            protein_name + "_all_singles.csv"
        data.save_all_singles(output_filename = mutations_location)
    else:
        args.mutations_location = \
            args.mutations_location + os.sep + \
            protein_name + ".csv"

    # Run inference
    evol_indices = model.compute_evol_indices(
        msa_data=data,
        list_mutations_location=args.mutations_location, 
        num_samples=args.num_samples_compute_evol_indices,
        batch_size=args.batch_size)
    list_valid_mutations, evol_indices, mean_elbo, std_elbo = evol_indices

    # Format as dataframe and write to file
    df = pd.DataFrame(dict(
        protein_name = protein_name,
        mutations = list_valid_mutations,
        evol_indices = evol_indices
    ))    
    evol_indices_output_filename = \
        args.output_evol_indices_location + os.sep + \
        protein_name + '_' + \
        str(args.num_samples_compute_evol_indices) + '_samples' + \
        args.output_evol_indices_filename_suffix + '.csv'
    try:
        keep_header = os.stat(evol_indices_output_filename).st_size == 0
    except:
        keep_header=True 
    df.to_csv(
        path_or_buf=evol_indices_output_filename, 
        index=False, mode='a', header=keep_header
        )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evol indices')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name is the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--computation_mode', type=str, help='Computes evol indices for all single AA mutations or for a passed in list of mutations (singles or multiples) [all_singles,input_mutations_list]')
    parser.add_argument('--all_singles_mutations_folder', type=str, help='Location for the list of generated single AA mutations')
    parser.add_argument('--mutations_location', type=str, help='Location of all mutations to compute the evol indices for')
    parser.add_argument('--output_evol_indices_location', type=str, help='Output location of computed evol indices')
    parser.add_argument('--output_evol_indices_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')
    parser.add_argument('--num_samples_compute_evol_indices', type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing evol indices')
    args = parser.parse_args()

    main(args)