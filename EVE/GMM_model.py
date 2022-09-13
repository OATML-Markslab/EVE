from sklearn import mixture
import numpy as np
import os
import tqdm

class GMM_model:
    """
    Class for the global-local GMM model trained on single-point mutations for a wild-type protein sequence
    """
    def __init__(self, params) -> None:
        # Set up logging
        self.log_location = params['log_location']
        if not os.path.exists(os.path.basename(self.log_location)):
            os.makedirs(os.path.basename(self.log_location))
        with open(self.log_location, "a") as logs:
            logs.write("protein_name,weight_pathogenic,mean_pathogenic,mean_benign,std_dev_pathogenic,std_dev_benign\n")
        
        # store parameters
        self.protein_GMM_weight = params['protein_GMM_weight']

    def fit_single(self, X_train, protein_name=None, verbose = False):
        model = mixture.GaussianMixture(
            n_components=2, 
            covariance_type='full',
            max_iter=1000,
            n_init=30,
            tol=1e-4
            )
        model.fit(X_train)
        #The pathogenic cluster is the cluster with higher mean value
        pathogenic_cluster_index = np.argmax(np.array(self.model.means_).flatten()) 

        with open(self.log_location, "a") as logs:
            logs.write(",".join(str(x) for x in [
                protein_name, 
                np.array(self.model.weights_).flatten()[self.pathogenic_cluster_index], 
                np.array(self.model.means_).flatten()[self.pathogenic_cluster_index],
                np.array(self.model.means_).flatten()[1 - self.pathogenic_cluster_index], 
                np.sqrt(np.array(self.model.covariances_).flatten()[self.pathogenic_cluster_index]),
                np.sqrt(np.array(self.model.covariances_).flatten()[1 - self.pathogenic_cluster_index])
            ])
            +"\n"
            )

        if verbose:
            inferred_params = self.model.get_params()
            print("Index of mixture component with highest mean: "+str(self.pathogenic_cluster_index))
            print("Model parameters: "+str(inferred_params))
            print("Mixture component weights: "+str(self.model.weights_))
            print("Mixture component means: "+str(self.model.means_))
            print("Cluster component cov: "+str(self.model.covariances_))
        
        return model, pathogenic_cluster_index
    
    def fit(self, X_train, proteins_train, verbose = True):
        # set up to train
        self.models = {}
        self.indices = {}

        # train global model
        gmm, index = self.fit_single(X_train,'main',verbose=verbose)
        self.models['main'] = gmm
        self.indices['main'] = index

        # train local models
        if self.protein_GMM_weight > 0.0:
            proteins_list = list(set(proteins_train))
            for protein in tqdm.tqdm(proteins_list, "Training all protein GMMs"):
                X_train_protein = X_train[proteins_train == protein]
                gmm, index = self.fit_single(X_train_protein,protein,verbose=verbose)
                self.models[protein] = gmm
                self.indices[protein] = index                

        return self.models, self.indices

    def predict(self, X_pred, protein):
        model = self.models[protein]
        cluster_index = self.indices[protein]
        scores = model.predict_proba(X_pred)[:,cluster_index]
        classes = (scores > 0.5).astype(int)
        return scores, classes

    def predict_weighted(self, X_pred, protein):      
        scores_protein = self.predict(X_pred, protein)
        scores_main = self.predict(X_pred, 'main')

        scores_weighted = scores_main * (1 - self.protein_GMM_weight) + \
            scores_protein * self.protein_GMM_weight
        classes_weighted = (scores_weighted > 0.5).astype(int)
        return scores_weighted, classes_weighted




X_train, groups_train = preprocess_indices(all_evol_indices)