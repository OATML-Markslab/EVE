import datetime
import os
import sys

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.special import erfinv
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils.data_utils import one_hot_3D, get_training_dataloader, get_one_hot_dataloader
from . import VAE_encoder, VAE_decoder


class VAE_model(nn.Module):
    """
    Class for the VAE model with estimation of weights distribution parameters via Mean-Field VI.
    """

    def __init__(self,
                 model_name,
                 data,
                 encoder_parameters,
                 decoder_parameters,
                 random_seed,
                 seq_len=None,
                 alphabet_size=None,
                 Neff=None,
                 ):

        super().__init__()

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.random_seed = random_seed
        torch.manual_seed(random_seed)

        self.seq_len = seq_len if seq_len is not None else data.seq_len
        self.alphabet_size = alphabet_size if alphabet_size is not None else data.alphabet_size
        self.Neff = Neff if Neff is not None else data.Neff

        self.encoder_parameters = encoder_parameters
        self.decoder_parameters = decoder_parameters

        encoder_parameters['seq_len'] = self.seq_len
        encoder_parameters['alphabet_size'] = self.alphabet_size
        decoder_parameters['seq_len'] = self.seq_len
        decoder_parameters['alphabet_size'] = self.alphabet_size

        self.encoder = VAE_encoder.VAE_MLP_encoder(params=encoder_parameters)
        if decoder_parameters['bayesian_decoder']:
            self.decoder = VAE_decoder.VAE_Bayesian_MLP_decoder(params=decoder_parameters)
        else:
            self.decoder = VAE_decoder.VAE_Standard_MLP_decoder(params=decoder_parameters)
        self.logit_sparsity_p = decoder_parameters['logit_sparsity_p']

    def sample_latent(self, mu, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu).to(self.device)
        z = torch.exp(0.5 * log_var) * eps + mu
        return z

    def KLD_diag_gaussians(self, mu, logvar, p_mu, p_logvar):
        """
        KL divergence between diagonal gaussian with prior diagonal gaussian.
        """
        KLD = 0.5 * (p_logvar - logvar) + 0.5 * (torch.exp(logvar) + torch.pow(mu - p_mu, 2)) / (
                torch.exp(p_logvar) + 1e-20) - 0.5

        return torch.sum(KLD)

    def annealing_factor(self, annealing_warm_up, training_step):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        if training_step < annealing_warm_up:
            return training_step / annealing_warm_up
        else:
            return 1

    def KLD_global_parameters(self):
        """
        KL divergence between the variational distributions and the priors (for the decoder weights).
        """
        KLD_decoder_params = 0.0
        zero_tensor = torch.tensor(0.0).to(self.device)

        for layer_index in range(len(self.decoder.hidden_layers_sizes)):
            for param_type in ['weight', 'bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                    self.decoder.state_dict(keep_vars=True)[
                        'hidden_layers_mean.' + str(layer_index) + '.' + param_type].flatten(),
                    self.decoder.state_dict(keep_vars=True)[
                        'hidden_layers_log_var.' + str(layer_index) + '.' + param_type].flatten(),
                    zero_tensor,
                    zero_tensor
                )

        for param_type in ['weight', 'bias']:
            KLD_decoder_params += self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['last_hidden_layer_' + param_type + '_mean'].flatten(),
                self.decoder.state_dict(keep_vars=True)['last_hidden_layer_' + param_type + '_log_var'].flatten(),
                zero_tensor,
                zero_tensor
            )

        if self.decoder.include_sparsity:
            self.logit_scale_sigma = 4.0
            self.logit_scale_mu = 2.0 ** 0.5 * self.logit_scale_sigma * erfinv(2.0 * self.logit_sparsity_p - 1.0)

            sparsity_mu = torch.tensor(self.logit_scale_mu).to(self.device)
            sparsity_log_var = torch.log(torch.tensor(self.logit_scale_sigma ** 2)).to(self.device)
            KLD_decoder_params += self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['sparsity_weight_mean'].flatten(),
                self.decoder.state_dict(keep_vars=True)['sparsity_weight_log_var'].flatten(),
                sparsity_mu,
                sparsity_log_var
            )

        if self.decoder.convolve_output:
            for param_type in ['weight']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                    self.decoder.state_dict(keep_vars=True)['output_convolution_mean.' + param_type].flatten(),
                    self.decoder.state_dict(keep_vars=True)['output_convolution_log_var.' + param_type].flatten(),
                    zero_tensor,
                    zero_tensor
                )

        if self.decoder.include_temperature_scaler:
            KLD_decoder_params += self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['temperature_scaler_mean'].flatten(),
                self.decoder.state_dict(keep_vars=True)['temperature_scaler_log_var'].flatten(),
                zero_tensor,
                zero_tensor
            )
        return KLD_decoder_params

    def loss_function(self, x_recon_log, x, mu, log_var, kl_latent_scale, kl_global_params_scale, annealing_warm_up,
                      training_step, Neff):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        """
        BCE = F.binary_cross_entropy_with_logits(x_recon_log, x, reduction='sum') / x.shape[0]
        KLD_latent = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.shape[0]
        if self.decoder.bayesian_decoder:
            KLD_decoder_params_normalized = self.KLD_global_parameters() / Neff
        else:
            KLD_decoder_params_normalized = 0.0
        warm_up_scale = self.annealing_factor(annealing_warm_up, training_step)
        neg_ELBO = BCE + warm_up_scale * (
                kl_latent_scale * KLD_latent + kl_global_params_scale * KLD_decoder_params_normalized)
        return neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized

    def all_likelihood_components(self, x):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        """
        mu, log_var = self.encoder(x)
        z = self.sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)

        recon_x_log = recon_x_log.view(-1, self.alphabet_size * self.seq_len)
        x = x.view(-1, self.alphabet_size * self.seq_len)

        BCE_batch_tensor = torch.sum(F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none'), dim=1)
        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        ELBO_batch_tensor = -(BCE_batch_tensor + KLD_batch_tensor)

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor

    def all_likelihood_components_z(self, x, mu, log_var):
        """Skip the encoder part and directly sample z"""
        # Need to run mu, log_var = self.encoder(x) first
        z = self.sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)

        recon_x_log = recon_x_log.view(-1, self.alphabet_size * self.seq_len)
        x = x.view(-1, self.alphabet_size * self.seq_len)

        BCE_batch_tensor = torch.sum(F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none'), dim=1)
        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        ELBO_batch_tensor = -(BCE_batch_tensor + KLD_batch_tensor)

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor

    def train_model(self, data, training_parameters, use_dataloader=False):
        """
        Training procedure for the VAE model.
        If use_validation_set is True then:
            - we split the alignment data in train/val sets.
            - we train up to num_training_steps steps but store the version of the model with lowest loss on validation set across training
        If not, then we train the model for num_training_steps and save the model at the end of training.
        
        use_dataloader: Whether to stream in the one-hot encodings via a dataloader. 
        If False, loads in the entire one-hot encoding matrix into memory and iterates over it.
        """
        if torch.cuda.is_available():
            cudnn.benchmark = True
        self.train()

        if training_parameters['log_training_info']:
            filename = training_parameters['training_logs_location'] + os.sep + self.model_name + "_losses.csv"
            with open(filename, "a") as logs:
                logs.write("Number of sequences in alignment file:\t" + str(data.num_sequences) + "\n")
                logs.write("Neff:\t" + str(self.Neff) + "\n")
                logs.write("Alignment sequence length:\t" + str(data.seq_len) + "\n")

        optimizer = optim.Adam(self.parameters(), lr=training_parameters['learning_rate'],
                               weight_decay=training_parameters['l2_regularization'])

        if training_parameters['use_lr_scheduler']:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_parameters['lr_scheduler_step_size'],
                                                  gamma=training_parameters['lr_scheduler_gamma'])

        list_sequences = list(data.seq_name_to_sequence.values())
        if training_parameters['use_validation_set']:
            if use_dataloader:
                seqs_train, seqs_val, weights_train, weights_val = train_test_split(list_sequences,
                                                                                    data.weights,
                                                                                    test_size=training_parameters['validation_set_pct'],
                                                                                    random_state=self.random_seed)
                # Validation still just passes in the whole one-hot encoding in one go
                x_val = one_hot_3D(seqs_val, alphabet=data.alphabet, seq_length=data.seq_len)
                assert len(seqs_train) == weights_train.shape[0]  # One weight per sequence
            else:
                x_train, x_val, weights_train, weights_val = train_test_split(data.one_hot_encoding, data.weights,
                                                                              test_size=training_parameters['validation_set_pct'],
                                                                              random_state=self.random_seed)
                assert x_train.shape[0] == weights_train.shape[0]  # One weight per sequence
            best_val_loss = float('inf')
            best_model_step_index = 0
        else:
            seqs_train = list_sequences
            weights_train = data.weights
            best_val_loss = None
            best_model_step_index = training_parameters['num_training_steps']
        
        seq_sample_probs = weights_train / np.sum(weights_train)
        
        # Keep old behaviour for comparison
        if use_dataloader:
            # Stream one-hot encodings
            train_dataloader = get_training_dataloader(sequences=seqs_train, weights=weights_train, alphabet=data.alphabet, seq_len=data.seq_len, batch_size=training_parameters['batch_size'], num_training_steps=training_parameters['num_training_steps'])
        else:
            batch_order = np.arange(x_train.shape[0])
            assert batch_order.shape == seq_sample_probs.shape, f"batch_order and seq_sample_probs must have the same shape. batch_order.shape={batch_order.shape}, seq_sample_probs.shape={seq_sample_probs.shape}"
            def get_mock_dataloader():
                while True:
                    # Sample a batch according to sequence weight
                    batch_index = np.random.choice(batch_order, training_parameters['batch_size'], p=seq_sample_probs).tolist()
                    batch = x_train[batch_index]
                    yield batch
            train_dataloader = get_mock_dataloader()

        self.Neff_training = np.sum(weights_train)

        start = time.time()
        train_loss = 0
        for training_step, batch in enumerate(tqdm(train_dataloader, desc="Training model", total=training_parameters['num_training_steps'], mininterval=5)):

            # For the dataloader, we may have to manually end training at
            if training_step >= training_parameters['num_training_steps']:
                break
            x = batch.to(self.device, dtype=self.dtype)
                
            optimizer.zero_grad()

            mu, log_var = self.encoder(x)
            z = self.sample_latent(mu, log_var)
            recon_x_log = self.decoder(z)

            neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = self.loss_function(
                recon_x_log, x, mu, log_var,
                training_parameters['kl_latent_scale'],
                training_parameters['kl_global_params_scale'],
                training_parameters['annealing_warm_up'],
                training_step,
                self.Neff_training)

            neg_ELBO.backward()
            optimizer.step()

            if training_parameters['use_lr_scheduler']:
                scheduler.step()

            if training_step % training_parameters['log_training_freq'] == 0:
                progress = "|Train : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, KLD_decoder_params_norm: {4:.3f}, Time: {5:.2f} |".format(
                    training_step, neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized, time.time() - start)
                print(progress)

                if training_parameters['log_training_info']:
                    with open(filename, "a+") as logs:
                        logs.write(progress + "\n")

            if training_step % training_parameters['save_model_params_freq'] == 0:
                self.save(model_checkpoint=training_parameters[
                                               'model_checkpoint_location'] + os.sep + self.model_name + "_step_" + str(
                    training_step),
                          encoder_parameters=self.encoder_parameters,
                          decoder_parameters=self.decoder_parameters,
                          training_parameters=training_parameters)

            if training_parameters['use_validation_set'] and training_step % training_parameters['validation_freq'] == 0:
                x_val = x_val.to(self.device, dtype=self.dtype)
                val_neg_ELBO, val_BCE, val_KLD_latent, val_KLD_global_parameters = self.test_model(x_val, weights_val,
                                                                                                   training_parameters[
                                                                                                       'batch_size'])

                progress_val = "\t\t\t|Val : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, KLD_decoder_params_norm: {4:.3f}, Time: {5:.2f} |".format(
                    training_step, val_neg_ELBO, val_BCE, val_KLD_latent, val_KLD_global_parameters,
                    time.time() - start)
                print(progress_val)
                if training_parameters['log_training_info']:
                    with open(filename, "a+") as logs:
                        logs.write(progress_val + "\n")

                if val_neg_ELBO < best_val_loss:
                    best_val_loss = val_neg_ELBO
                    best_model_step_index = training_step
                    self.save(model_checkpoint=training_parameters[
                                                   'model_checkpoint_location'] + os.sep + self.model_name + "_best",
                              encoder_parameters=self.encoder_parameters,
                              decoder_parameters=self.decoder_parameters,
                              training_parameters=training_parameters)
                self.train()
        
        
    def test_model(self, x_val, weights_val, batch_size):
        self.eval()

        with torch.no_grad():
            val_batch_order = np.arange(x_val.shape[0])
            val_seq_sample_probs = weights_val / np.sum(weights_val)

            val_batch_index = np.random.choice(val_batch_order, batch_size, p=val_seq_sample_probs).tolist()
            x = x_val[val_batch_index].to(self.device, dtype=self.dtype)
            mu, log_var = self.encoder(x)
            z = self.sample_latent(mu, log_var)
            recon_x_log = self.decoder(z)

            neg_ELBO, BCE, KLD_latent, KLD_global_parameters = self.loss_function(recon_x_log, x, mu, log_var,
                                                                                  kl_latent_scale=1.0,
                                                                                  kl_global_params_scale=1.0,
                                                                                  annealing_warm_up=0, training_step=1,
                                                                                  Neff=self.Neff_training)  # set annealing factor to 1

        return neg_ELBO.item(), BCE.item(), KLD_latent.item(), KLD_global_parameters.item()

    def save(self, model_checkpoint, encoder_parameters, decoder_parameters, training_parameters, batch_size=256):
        # Create intermediate dirs above this
        os.makedirs(os.path.dirname(model_checkpoint), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'encoder_parameters': encoder_parameters,
            'decoder_parameters': decoder_parameters,
            'training_parameters': training_parameters,
        }, model_checkpoint)

    def compute_evol_indices(self, msa_data, list_mutations_location, num_samples, batch_size=256,
                             mutant_column="mutations"):
        """
            The column in the list_mutations dataframe that contains the mutant(s) for a given variant should be called "mutations"
        """

        # Note: wt is added inside this function, so no need to add a row in csv/dataframe input with wt
        list_mutations = pd.read_csv(list_mutations_location, header=0)

        # Multiple mutations are to be passed colon-separated
        # Remove (multiple) mutations that are invalid
        list_valid_mutations, list_valid_mutated_sequences = self.validate_mutants(msa_data=msa_data, mutations=list_mutations[mutant_column])

        # first sequence in the list is the wild_type
        list_valid_mutations = ['wt'] + list_valid_mutations
        list_valid_mutated_sequences['wt'] = msa_data.focus_seq_trimmed

        dataloader = get_one_hot_dataloader(seq_keys=list_valid_mutations,
                                            seq_name_to_sequence=list_valid_mutated_sequences,
                                            alphabet=msa_data.alphabet,
                                            seq_len=len(msa_data.focus_cols),
                                            batch_size=batch_size)

        # Store wt_mean_predictions
        with torch.no_grad():
            mean_predictions = torch.zeros(len(list_valid_mutations))
            std_predictions = torch.zeros(len(list_valid_mutations))
            for i, batch in enumerate(tqdm(dataloader, 'Looping through mutation batches')):
                batch_samples = torch.zeros(len(batch), num_samples, dtype=self.dtype, device=self.device)  # Keep this on GPU
                x = batch.type(self.dtype).to(self.device)
                mu, log_var = self.encoder(x)
                for j in tqdm(range(num_samples), 'Looping through number of samples for batch #: ' + str(i + 1), mininterval=5):
                    # seq_predictions, _, _ = self.all_likelihood_components(x)
                    seq_predictions, _, _ = self.all_likelihood_components_z(x, mu, log_var)
                    batch_samples[:, j] = seq_predictions  # Note: We could move this straight to CPU to save GPU space

                mean_predictions[i * batch_size:i * batch_size + len(x)] = batch_samples.mean(dim=1, keepdim=False)
                std_predictions[i * batch_size:i * batch_size + len(x)] = batch_samples.std(dim=1, keepdim=False)
                tqdm.write('\n')

            delta_elbos = mean_predictions - mean_predictions[0]
            evol_indices = - delta_elbos.detach().cpu().numpy()

        return list_valid_mutations, evol_indices, mean_predictions[0].detach().cpu().numpy(), std_predictions.detach().cpu().numpy()

    def validate_mutants(self, msa_data, mutations):
        list_valid_mutations = []
        list_valid_mutated_sequences = {}

        for mutation in mutations:
            try:
                individual_substitutions = str(mutation).split(':')
            except Exception as e:
                print("Error with mutant {}".format(str(mutation)))
                print("Specific error: " + str(e))
                continue
            mutated_sequence = list(msa_data.focus_seq_trimmed)[:]
            fully_valid_mutation = True
            for mut in individual_substitutions:
                try:
                    wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
                    if wt_aa == mut_aa: # Skip synonymous
                        continue
                    # Log specific invalid mutants
                    if pos not in msa_data.uniprot_focus_col_to_wt_aa_dict:
                        print("pos {} not in uniprot_focus_col_to_wt_aa_dict".format(pos))
                        fully_valid_mutation = False
                    # Given it's in the dict, check if it's a valid mutation
                    elif msa_data.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa:
                        print("wt_aa {} != uniprot_focus_col_to_wt_aa_dict[{}] {}".format(
                            wt_aa, pos, msa_data.uniprot_focus_col_to_wt_aa_dict[pos]))
                        fully_valid_mutation = False
                    if mut not in msa_data.mutant_to_letter_pos_idx_focus_list:
                        print("mut {} not in mutant_to_letter_pos_idx_focus_list".format(mut))
                        fully_valid_mutation = False

                    if fully_valid_mutation:
                        wt_aa, pos, idx_focus = msa_data.mutant_to_letter_pos_idx_focus_list[mut]
                        mutated_sequence[idx_focus] = mut_aa  # perform the corresponding AA substitution
                    else:
                        print("Not a valid mutant: " + mutation)
                        break

                except Exception as e:
                    print("Error processing mutation {} in mutant {}".format(str(mut), str(mutation)))
                    print("Specific error: " + str(e))
                    fully_valid_mutation = False
                    break

            if fully_valid_mutation:
                list_valid_mutations.append(mutation)
                list_valid_mutated_sequences[mutation] = ''.join(mutated_sequence)

        return list_valid_mutations, list_valid_mutated_sequences
