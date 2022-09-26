from sklearn import mixture
import numpy as np
import os
import tqdm
import logging

class GMM_model:
    """
    Class for the global-local GMM model trained on single-point mutations for a wild-type protein sequence
    """
    def __init__(self, params) -> None:
              
        # store parameters
        self.protein_GMM_weight = params['protein_GMM_weight']
        self.params = dict(
            n_components=2, 
            covariance_type='full',
            max_iter=1000,
            n_init=30,
            tol=1e-4
        )

    def fit_single(self, X_train):
        model = mixture.GaussianMixture(**self.params)
        model.fit(X_train)
        #The pathogenic cluster is the cluster with higher mean value
        pathogenic_cluster_index = np.argmax(np.array(model.means_).flatten()) 
        return model, pathogenic_cluster_index
    
    def fit(self, X_train, proteins_train):
        # set up to train
        self.models = {}
        self.indices = {}

        # train global model
        gmm, index = self.fit_single(X_train,'main')
        self.models['main'] = gmm
        self.indices['main'] = index

        # train local models
        if self.protein_GMM_weight > 0.0:
            proteins_list = list(set(proteins_train))
            for protein in tqdm.tqdm(proteins_list, "Training all protein GMMs"):
                X_train_protein = X_train[proteins_train == protein]
                gmm, index = self.fit_single(X_train_protein,protein)
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

    def get_fitted_params(self):
        fitted_params = (
            self.models.keys(), 
            self.models.values(), 
            self.indices.values()
            )
        output = {}
        for protein, model, index in zip(*fitted_params):
            output[protein] = {
                'index': index, 
                'means': model.means_,
                'covar': model.covariances_,
                'weights': model.weights_
            }
        return output
        
